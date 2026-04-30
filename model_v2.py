"""
model_v2.py — Dispositional Prediction Model with Counterfactual Speaker Modeling.

Extension of DispositionalPredictionModel that adds a counterfactual gate
to the speaker context phase.

At turn t, when predicting speaker A's emotion:
    c_actual       = A's context built from A's own past utterances
    c_counterfactual = GRU_step(c_actual, δu_{t-1})  ← "what if A had said what B just said?"
    gate           = σ(W · cat(c_actual, c_cf, δu_{t-1}))
    c_blended      = gate ⊙ c_actual + (1 - gate) ⊙ c_cf

    gate → 1 : A is emotionally AUTONOMOUS (own trajectory dominates)
    gate → 0 : A is emotionally REACTIVE  (B's utterance shapes A's state)

The gate is interpretable per-turn and per-speaker, and can be analyzed
to distinguish reactive vs autonomous speakers across conversations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from config import ModelConfig
from model import DispositionalPredictionModel, MAX_LOCAL_SPEAKERS


# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual Speaker Gate
# ──────────────────────────────────────────────────────────────────────────────

class CounterfactualSpeakerGate(nn.Module):
    """
    Blends a speaker's actual context with a counterfactual context
    ("what would my state be if I had said what you just said?").

    gate  = σ(W · cat(c_actual, c_cf, δu_prev))
    out   = gate ⊙ c_actual + (1 - gate) ⊙ c_cf

    gate close to 1 → speaker is emotionally autonomous
    gate close to 0 → speaker is emotionally reactive / contagious
    """

    def __init__(self, speaker_context_dim: int, perturbation_dim: int):
        super().__init__()
        in_dim = speaker_context_dim * 2 + perturbation_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, speaker_context_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[0].bias)
        nn.init.xavier_uniform_(self.gate[0].weight)

    def forward(
        self,
        c_actual: torch.Tensor,   # (B, ctx_dim) — real speaker context
        c_cf:     torch.Tensor,   # (B, ctx_dim) — counterfactual context
        du_prev:  torch.Tensor,   # (B, pd)       — other speaker's last utterance
    ):
        g = self.gate(torch.cat([c_actual, c_cf, du_prev], dim=-1))   # (B, ctx_dim)
        return g * c_actual + (1.0 - g) * c_cf, g


# ──────────────────────────────────────────────────────────────────────────────
# Extended model
# ──────────────────────────────────────────────────────────────────────────────

class DispositionalPredictionModelV2(DispositionalPredictionModel):
    """
    Inherits all components from DispositionalPredictionModel.
    Adds CounterfactualSpeakerGate in Phase 2 (speaker context building).
    Everything else — LLaMA encoder, CausalTransformerDynamics,
    prediction heads, staged-training helpers — is unchanged.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.cf_gate = CounterfactualSpeakerGate(
            cfg.speaker_context_dim, cfg.perturbation_dim
        )

    def forward(self, batch: Dict, delta_u_cache: Optional[Dict] = None) -> Dict:
        B, T, L = batch["input_ids"].shape
        device   = batch["input_ids"].device

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        speaker_ids    = batch["speaker_ids"]
        emotion_ids    = batch["emotion_ids"]
        lengths        = batch["length"]
        valid_mask     = (emotion_ids >= 0)

        # ── Phase 1: LLaMA encode (identical to parent) ───────────────────────
        dial_ids = batch.get("dialogue_id", None)
        if (delta_u_cache is not None and dial_ids is not None and
                all(d in delta_u_cache for d in dial_ids)):
            flat_pooled = torch.stack(
                [delta_u_cache[d].to(device) for d in dial_ids], dim=0
            ).view(B * T, -1)
            flat_delta  = self.perturbation_enc.proj(flat_pooled.float())
            all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim)
        else:
            flat_ids    = input_ids.view(B * T, L)
            flat_mask   = attention_mask.view(B * T, L)
            flat_hidden = self.llama_encoder.encode(flat_ids, flat_mask)
            flat_delta  = self.perturbation_enc(flat_hidden, flat_mask)
            all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim)
            del flat_ids, flat_mask, flat_hidden, flat_delta

        # ── Phase 2: speaker context + counterfactual gate ────────────────────
        spk_ctx      = self.speaker_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        label_ctx    = self.label_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        spk_last_emo = torch.full((B, MAX_LOCAL_SPEAKERS), self.cfg.num_emotions,
                                  dtype=torch.long, device=device)

        all_spk_ctx      = []
        all_label_ctx    = []
        all_prev_spk_emo = []
        all_gates        = []   # (B, T, ctx_dim) autonomy scores

        for t in range(T):
            spk_t   = speaker_ids[:, t]
            valid_t = valid_mask[:, t]
            spk_idx = spk_t.clamp(max=MAX_LOCAL_SPEAKERS - 1)

            c_actual = self.speaker_context.get_context(spk_ctx, spk_t)  # (B, ctx_dim)

            if t > 0:
                # Counterfactual: run previous utterance through active speaker's GRU
                # "what would A's state be if A had said what B just said?"
                du_prev = all_delta_u[:, t - 1]                              # (B, pd)
                c_cf    = self.speaker_context.gru(du_prev.float(), c_actual.float())
                c_blended, gate = self.cf_gate(c_actual, c_cf, du_prev)
            else:
                # t=0: no prior utterance — use actual context, gate = 1 (autonomous)
                c_blended = c_actual
                gate      = torch.ones(B, self.cfg.speaker_context_dim, device=device)

            all_spk_ctx.append(c_blended)
            all_label_ctx.append(self.label_context.get_context(label_ctx, spk_t))
            all_prev_spk_emo.append(spk_last_emo[torch.arange(B, device=device), spk_idx])
            all_gates.append(gate)

            if t < T - 1:
                spk_ctx_new = self.speaker_context.update(spk_ctx, spk_t, all_delta_u[:, t])
                spk_ctx = torch.where(
                    valid_t.unsqueeze(-1).unsqueeze(-1), spk_ctx_new, spk_ctx
                )
                label_ctx_new = self.label_context.update(label_ctx, spk_t, emotion_ids[:, t])
                label_ctx = torch.where(
                    valid_t.unsqueeze(-1).unsqueeze(-1), label_ctx_new, label_ctx
                )
                eid_t   = emotion_ids[:, t]
                cur_emo = spk_last_emo[torch.arange(B, device=device), spk_idx]
                new_emo = torch.where(valid_t, eid_t.clamp(min=0), cur_emo)
                spk_last_emo[torch.arange(B, device=device), spk_idx] = new_emo

        all_spk_ctx      = torch.stack(all_spk_ctx,      dim=1)   # (B, T, cd)
        all_label_ctx    = torch.stack(all_label_ctx,    dim=1)   # (B, T, lc)
        all_prev_spk_emo = torch.stack(all_prev_spk_emo, dim=1)   # (B, T)
        all_gates        = torch.stack(all_gates,        dim=1)   # (B, T, cd)

        # ── Phase 3: causal transformer (identical to parent) ─────────────────
        disp_states = self.personal_dynamics(
            all_delta_u, all_spk_ctx, all_label_ctx, all_prev_spk_emo, valid_mask, speaker_ids
        )

        # ── Predictions (identical to parent) ────────────────────────────────
        s_flat  = disp_states.view(B * T, -1)
        du_flat = all_delta_u.view(B * T, -1)

        prior_logits = self.prediction_head(s_flat).view(B, T, -1)
        recog_logits = self.recognition_head(du_flat.detach()).view(B, T, -1)

        p  = F.softmax(prior_logits, dim=-1).clamp(min=1e-9)
        kl = F.kl_div(
            p[:, :-1].log(), p[:, 1:].clamp(min=1e-9), reduction="none"
        ).sum(-1)
        surprise = torch.cat(
            [torch.zeros(B, 1, device=device), kl], dim=1
        ) * valid_mask.float()

        return {
            "prediction_logits":    prior_logits,
            "recognition_logits":   recog_logits,
            "dispositional_states": disp_states,
            "surprise":             surprise,
            "speaker_contexts":     all_spk_ctx,
            "speaker_ids":          speaker_ids,
            "emotion_ids":          emotion_ids,
            "lengths":              lengths,
            # gate: mean over ctx_dim → per-turn autonomy scalar in [0,1]
            "cf_gates":             all_gates.mean(dim=-1),   # (B, T)
        }
