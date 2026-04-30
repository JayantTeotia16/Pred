"""
model_v3.py — Ablation model with four optional additions.

Each flag is independently togglable:

  use_cross_speaker_emo  : feed other-speaker's last emotion into DynamicSpeakerContext GRU
  use_future_pred        : auxiliary head s(t) → emotion[t+1], weight=future_pred_weight
  use_joint_transition   : auxiliary BCE head s(t) → P(emotion changes at t), joint training

Change 1 (recognition_loss_weight=0 + no staged) is a config-only change —
no model modification needed; handled entirely in test_ablation_runner.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from config import ModelConfig
from model import DispositionalPredictionModel, MAX_LOCAL_SPEAKERS
from dispositional_module import PredictionHead


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Speaker Emotion Context
# ──────────────────────────────────────────────────────────────────────────────

class CrossSpeakerEmotionContext(nn.Module):
    """
    Drop-in replacement for DynamicSpeakerContext.
    GRU input: cat(δu, embed(other_speaker_last_emotion))

    "Other speaker's last emotion" = last emotion seen in the conversation
    from any speaker other than the currently active one. For dyadic
    conversations (IEMOCAP) this is exactly the partner's last emotion.
    The embedding dimension reuses emotion_label_embed_dim.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.context_dim    = cfg.speaker_context_dim
        self.num_emotions   = cfg.num_emotions
        self.other_emo_embed = nn.Embedding(cfg.num_emotions + 1, cfg.emotion_label_embed_dim)
        self.gru = nn.GRUCell(
            input_size=cfg.perturbation_dim + cfg.emotion_label_embed_dim,
            hidden_size=cfg.speaker_context_dim,
        )
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def init_states(self, B: int, max_speakers: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, max_speakers, self.context_dim, device=device)

    def get_context(self, states: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        B   = states.shape[0]
        idx = speaker_ids.clamp(max=states.shape[1] - 1)
        return states[torch.arange(B, device=states.device), idx]

    def update(
        self,
        states:        torch.Tensor,  # (B, S, context_dim)
        speaker_ids:   torch.Tensor,  # (B,)
        delta_u:       torch.Tensor,  # (B, perturbation_dim)
        other_emo_ids: torch.Tensor,  # (B,) long — other speaker's last emotion
    ) -> torch.Tensor:
        B, S, _ = states.shape
        idx      = speaker_ids.clamp(max=S - 1)
        h_old    = states[torch.arange(B, device=states.device), idx]
        other_emb = self.other_emo_embed(other_emo_ids)
        inp      = torch.cat([delta_u.float(), other_emb.float()], dim=-1)
        h_new    = self.gru(inp, h_old.float())
        states_new = states.clone()
        states_new[torch.arange(B, device=states.device), idx] = h_new.to(states.dtype)
        return states_new


# ──────────────────────────────────────────────────────────────────────────────
# V3 model
# ──────────────────────────────────────────────────────────────────────────────

class DispositionalPredictionModelV3(DispositionalPredictionModel):
    """
    DispositionalPredictionModel with three independently togglable additions.
    Forward is identical to parent when all flags are False.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        use_cross_speaker_emo: bool  = False,
        use_future_pred:       bool  = False,
        use_joint_transition:  bool  = False,
    ):
        super().__init__(cfg)
        self.use_cross_speaker_emo = use_cross_speaker_emo
        self.use_future_pred       = use_future_pred
        self.use_joint_transition  = use_joint_transition

        if use_cross_speaker_emo:
            self.speaker_context = CrossSpeakerEmotionContext(cfg)

        if use_future_pred:
            self.future_pred_head = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions)

        if use_joint_transition:
            self.transition_head = nn.Sequential(
                nn.Linear(cfg.dispositional_state_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict, delta_u_cache: Optional[Dict] = None) -> Dict:
        if self.use_cross_speaker_emo:
            return self._forward_cross_speaker(batch, delta_u_cache)
        # No cross-speaker injection → run parent forward then augment outputs
        outputs = super().forward(batch, delta_u_cache)
        self._add_aux_outputs(outputs)
        return outputs

    def _add_aux_outputs(self, outputs: Dict):
        """Append future_pred and/or transition outputs to an existing forward output dict."""
        disp = outputs["dispositional_states"]   # (B, T, D)
        emos = outputs["emotion_ids"]             # (B, T)
        B, T, D = disp.shape

        if self.use_future_pred:
            # s(t) → emotion[t+1] for t in 0..T-2
            s_flat = disp[:, :-1].reshape(B * (T - 1), D)
            fut_logits = self.future_pred_head(s_flat).view(B, T - 1, -1)
            fut_labels = emos[:, 1:]
            fut_valid  = (emos[:, :-1] >= 0) & (emos[:, 1:] >= 0)
            outputs["future_logits"] = fut_logits
            outputs["future_labels"] = fut_labels
            outputs["future_valid"]  = fut_valid

        if self.use_joint_transition:
            # s(t) → P(emotion[t] != emotion[t-1]) for t in 1..T-1
            s_flat = disp[:, 1:].reshape(B * (T - 1), D)
            trans_logits = self.transition_head(s_flat).squeeze(-1).view(B, T - 1)
            trans_valid  = (emos[:, :-1] >= 0) & (emos[:, 1:] >= 0)
            trans_labels = (
                (emos[:, 1:] != emos[:, :-1]) & trans_valid
            ).float()
            outputs["transition_logits"] = trans_logits
            outputs["transition_labels"] = trans_labels
            outputs["transition_valid"]  = trans_valid

    def _forward_cross_speaker(
        self, batch: Dict, delta_u_cache: Optional[Dict] = None
    ) -> Dict:
        """
        Full forward with cross-speaker emotion injection in the GRU loop.
        Tracks last_conv_emo: the most recent emotion from the conversation
        (= partner's last emotion in dyadic/alternating conversations).
        """
        from model import LLaMAEncoder  # avoid circular import noise
        B, T, L = batch["input_ids"].shape
        device  = batch["input_ids"].device

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        speaker_ids    = batch["speaker_ids"]
        emotion_ids    = batch["emotion_ids"]
        lengths        = batch["length"]
        valid_mask     = (emotion_ids >= 0)

        # ── Phase 1: encode ────────────────────────────────────────────────
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

        # ── Phase 2: GRU loop with cross-speaker emotion ───────────────────
        spk_ctx      = self.speaker_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        label_ctx    = self.label_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        spk_last_emo = torch.full((B, MAX_LOCAL_SPEAKERS), self.cfg.num_emotions,
                                  dtype=torch.long, device=device)
        # Last emotion seen from any OTHER speaker (proxy for partner emotion in dyadic)
        last_conv_emo = torch.full((B,), self.cfg.num_emotions, dtype=torch.long, device=device)

        all_spk_ctx, all_label_ctx, all_prev_spk_emo = [], [], []

        for t in range(T):
            spk_t   = speaker_ids[:, t]
            valid_t = valid_mask[:, t]
            spk_idx = spk_t.clamp(max=MAX_LOCAL_SPEAKERS - 1)

            all_spk_ctx.append(self.speaker_context.get_context(spk_ctx, spk_t))
            all_label_ctx.append(self.label_context.get_context(label_ctx, spk_t))
            all_prev_spk_emo.append(spk_last_emo[torch.arange(B, device=device), spk_idx])

            if t < T - 1:
                other_emo_t = last_conv_emo.clone()

                spk_ctx_new = self.speaker_context.update(
                    spk_ctx, spk_t, all_delta_u[:, t], other_emo_t
                )
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

                # Update last_conv_emo: the emotion at this turn becomes the "other" for next turn
                new_last = eid_t.clamp(min=0)
                last_conv_emo = torch.where(valid_t, new_last, last_conv_emo)

        all_spk_ctx      = torch.stack(all_spk_ctx, dim=1)
        all_label_ctx    = torch.stack(all_label_ctx, dim=1)
        all_prev_spk_emo = torch.stack(all_prev_spk_emo, dim=1)

        # ── Phase 3: causal transformer ────────────────────────────────────
        disp_states = self.personal_dynamics(
            all_delta_u, all_spk_ctx, all_label_ctx, all_prev_spk_emo, valid_mask, speaker_ids
        )

        s_flat  = disp_states.view(B * T, -1)
        du_flat = all_delta_u.view(B * T, -1)

        prior_logits = self.prediction_head(s_flat).view(B, T, -1)
        recog_logits = self.recognition_head(du_flat.detach()).view(B, T, -1)

        p  = F.softmax(prior_logits, dim=-1).clamp(min=1e-9)
        kl = F.kl_div(p[:, :-1].log(), p[:, 1:].clamp(min=1e-9), reduction="none").sum(-1)
        surprise = torch.cat([torch.zeros(B, 1, device=device), kl], dim=1) * valid_mask.float()

        outputs = {
            "prediction_logits":    prior_logits,
            "recognition_logits":   recog_logits,
            "dispositional_states": disp_states,
            "surprise":             surprise,
            "speaker_contexts":     all_spk_ctx,
            "speaker_ids":          speaker_ids,
            "emotion_ids":          emotion_ids,
            "lengths":              lengths,
        }
        self._add_aux_outputs(outputs)
        return outputs

    # ── Helpers ────────────────────────────────────────────────────────────

    def rebuild_prediction_head(self, num_emotions: int):
        super().rebuild_prediction_head(num_emotions)
        if self.use_future_pred:
            self.future_pred_head = PredictionHead(self.cfg.dispositional_state_dim, num_emotions)
