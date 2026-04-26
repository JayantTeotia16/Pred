"""
model.py — Universal Dispositional Prediction Model.

Forward pass (per turn t):

    BEFORE seeing utterance t:
        c_t  = speaker_context[active_speaker]   ← built from their past turns
        s(t) = ODE_step(s(t-1), c_t, δu_{t-1})  ← uses PAST perturbation
        logits_t = PredictionHead(s(t), scene_s) ← pure prior, zero leakage

    AFTER storing the prediction:
        hidden_t = LLaMA(utterance_t)            ← encode current utterance
        δu_t     = PerturbationEncoder(hidden_t)
        scene_influence = SceneDynamics(scene_s, s(t))
        δu_t_final = δu_t + scene_influence
        speaker_context = GRU_update(speaker_context, δu_t_final, active_speaker)
        → ready for next turn

No global speaker vocabulary. All speaker personalisation is
derived from what that speaker has said in THIS conversation.
"""

import torch
import torch.nn as nn

from typing import Dict, List, Optional

from transformers import AutoModel
from config import ModelConfig
from dispositional_module import (
    PerturbationEncoder,
    DynamicSpeakerContext,
    EmotionLabelContext,
    PredictiveCodingEmotionModule,
    SceneDynamicsField,
    PredictionHead,
    SIGReg,
)


# ──────────────────────────────────────────────────────────────────────────────
# LLaMA encoder (frozen or LoRA-adapted)
# ──────────────────────────────────────────────────────────────────────────────

class LLaMAEncoder(nn.Module):
    """
    LLaMA feature extractor for past utterances.
    With use_lora=True (default): lightweight LoRA adapters on q/v projections
    allow the encoder to adapt to the emotion domain while keeping memory low.
    With use_lora=False: fully frozen, no gradients through LLaMA.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        print(f"  Loading LLaMA: {cfg.llama_model_name}")
        self.llama = AutoModel.from_pretrained(
            cfg.llama_model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16,
        )
        self.use_lora = cfg.use_lora

        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.llama = get_peft_model(self.llama, lora_cfg)
            # Required for gradient checkpointing to work with PEFT
            self.llama.enable_input_require_grads()
            self.llama.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in self.llama.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.llama.parameters())
            print(f"  LoRA applied — trainable: {trainable:,} / {total:,} params")
            print(f"  Gradient checkpointing enabled.")
        else:
            for p in self.llama.parameters():
                p.requires_grad = False
            print("  LLaMA frozen.")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns last-layer hidden states (B, L, H) in fp32."""
        if self.use_lora:
            out = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        return out.hidden_states[-1].float()


# keep old name as alias so existing checkpoints load without errors
FrozenLLaMAEncoder = LLaMAEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

# Maximum local speakers we track per conversation.
# Conversations with more speakers reuse slots (rare edge case).
MAX_LOCAL_SPEAKERS = 16


class DispositionalPredictionModel(nn.Module):
    """
    Universal dispositional emotion prediction model.

    Works with any conversational emotion dataset.
    Requires no global speaker vocabulary.
    Adapts to any number of emotion classes (set from DataConfig at load time).

    The dispositional state s(t) is built purely from the conversation's
    own history — no pre-trained speaker identity assumption.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.llama_encoder    = LLaMAEncoder(cfg)
        self.perturbation_enc = PerturbationEncoder(cfg.llama_hidden_size, cfg.perturbation_dim)
        self.speaker_context  = DynamicSpeakerContext(cfg)
        self.label_context    = EmotionLabelContext(cfg)
        self.pcem             = PredictiveCodingEmotionModule(cfg)
        self.scene_dynamics   = SceneDynamicsField(cfg) if cfg.use_scene_dynamics else None

        scene_dim = cfg.scene_state_dim if cfg.use_scene_dynamics else 0

        # Prior head — applied to belief BEFORE utterance t (no leakage by design)
        self.prediction_head = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions, scene_dim)

        # Auxiliary: multi-step future prediction heads (training only)
        self.future_head_1   = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions, scene_dim)
        self.future_head_2   = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions, scene_dim)

        # Posterior head — applied to belief AFTER precision-weighted error correction
        self.posterior_head  = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions, scene_dim)

        # SIGReg — Gaussian regulariser on belief state space
        self.sigreg          = SIGReg(n_projections=256)


    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict, delta_u_cache: Optional[Dict] = None) -> Dict:
        """
        Three-phase forward pass:
          Phase 1 — batched LLaMA encode + perturbation (B*T, L)
          Phase 2 — GRU speaker context pass (fast, T steps)
          Phase 3 — causal transformer → dispositional states (batched)
          Phase 4 — lightweight turn loop for scene dynamics + surprise

        Strictly causal: predicting emotion at turn t uses only turns 0..t-1.
        """
        B, T, L  = batch["input_ids"].shape
        device   = batch["input_ids"].device

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        speaker_ids    = batch["speaker_ids"]
        emotion_ids    = batch["emotion_ids"]
        lengths        = batch["length"]
        valid_mask     = (emotion_ids >= 0)           # (B, T) bool

        # ── Phase 1: encode all utterances in one LLaMA call ─────────────
        # If a delta_u cache is provided (frozen-LoRA phases), skip 8B forward
        dial_ids = batch.get("dialogue_id", None)
        if (delta_u_cache is not None and dial_ids is not None and
                all(d in delta_u_cache for d in dial_ids)):
            # Cache stores pooled LLaMA hiddens (B, T, H_llama) — still run proj MLP
            flat_pooled = torch.stack(
                [delta_u_cache[d].to(device) for d in dial_ids], dim=0
            ).view(B * T, -1)                                              # (B*T, H_llama)
            flat_delta  = self.perturbation_enc.proj(flat_pooled.float())  # (B*T, pd)
            all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim) # (B, T, pd)
        else:
            flat_ids    = input_ids.view(B * T, L)
            flat_mask   = attention_mask.view(B * T, L)
            flat_hidden = self.llama_encoder.encode(flat_ids, flat_mask)   # (B*T, L, H)
            flat_delta  = self.perturbation_enc(flat_hidden, flat_mask)    # (B*T, pd)
            all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim) # (B, T, pd)
            del flat_ids, flat_mask, flat_hidden, flat_delta

        # ── Phase 2: build speaker contexts turn-by-turn (GRU pass) ──────
        spk_ctx      = self.speaker_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        label_ctx    = self.label_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        # Tracks the last observed emotion per speaker (unknown = num_emotions token)
        spk_last_emo = torch.full((B, MAX_LOCAL_SPEAKERS), self.cfg.num_emotions,
                                  dtype=torch.long, device=device)
        all_spk_ctx      = []
        all_label_ctx    = []
        all_prev_spk_emo = []

        for t in range(T):
            spk_t   = speaker_ids[:, t]
            valid_t = valid_mask[:, t]
            spk_idx = spk_t.clamp(max=MAX_LOCAL_SPEAKERS - 1)

            all_spk_ctx.append(self.speaker_context.get_context(spk_ctx, spk_t))
            all_label_ctx.append(self.label_context.get_context(label_ctx, spk_t))
            all_prev_spk_emo.append(spk_last_emo[torch.arange(B, device=device), spk_idx])

            if t < T - 1:
                spk_ctx_new = self.speaker_context.update(spk_ctx, spk_t, all_delta_u[:, t])
                spk_ctx = torch.where(
                    valid_t.unsqueeze(-1).unsqueeze(-1), spk_ctx_new, spk_ctx
                )
                label_ctx_new = self.label_context.update(label_ctx, spk_t, emotion_ids[:, t])
                label_ctx = torch.where(
                    valid_t.unsqueeze(-1).unsqueeze(-1), label_ctx_new, label_ctx
                )
                # Update per-speaker last-known emotion (only for valid turns)
                eid_t   = emotion_ids[:, t]
                cur_emo = spk_last_emo[torch.arange(B, device=device), spk_idx]
                new_emo = torch.where(valid_t, eid_t.clamp(min=0), cur_emo)
                spk_last_emo[torch.arange(B, device=device), spk_idx] = new_emo

        all_spk_ctx      = torch.stack(all_spk_ctx, dim=1)       # (B, T, cd)
        all_label_ctx    = torch.stack(all_label_ctx, dim=1)     # (B, T, lc)
        all_prev_spk_emo = torch.stack(all_prev_spk_emo, dim=1)  # (B, T) long

        # ── Phase 3: PCEM → prior beliefs, posterior beliefs, prediction errors ──
        # prior_beliefs[t]  = b_s(t): belief BEFORE utterance t  (strictly causal)
        # post_beliefs[t]   = b_s(t) + π⊙W_e·ε_t: belief AFTER error correction
        # errors[t]         = actual_δu_t − G(b_s(t)): prediction error
        prior_beliefs, post_beliefs, errors = self.pcem(
            all_delta_u, all_spk_ctx, all_label_ctx, all_prev_spk_emo, valid_mask, speaker_ids
        )

        # Reconstruction loss — generator in PCEM predicts δu from belief.
        # Kept for compatibility (jepa_loss_weight=0 in config, no gradient effect).
        recon_mse = errors.float().pow(2).mean(-1)                              # (B, T)
        jepa_loss = (recon_mse * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        # ── Phase 4a: scene states — sequential, each step depends on prev ──────
        if self.scene_dynamics is not None:
            scene_list = []
            scene_s    = self.scene_dynamics.initial_state(B, device)
            for t in range(T):
                scene_list.append(scene_s)
                if t < T - 1:
                    scene_s_new = self.scene_dynamics.step(scene_s, prior_beliefs[:, t])
                    scene_s = torch.where(
                        valid_mask[:, t].unsqueeze(-1), scene_s_new, scene_s
                    )
            scene_states = torch.stack(scene_list, dim=1)   # (B, T, Ds)
        else:
            scene_states = None

        # ── Phase 4b: batched predictions ────────────────────────────────────
        s_prior_flat = prior_beliefs.view(B * T, -1)                     # (B*T, D)
        s_post_flat  = post_beliefs.view(B * T, -1)                      # (B*T, D)
        sc_flat      = scene_states.view(B * T, -1) if scene_states is not None else None

        prior_logits = self.prediction_head(s_prior_flat, sc_flat).view(B, T, -1)
        fut1_logits  = self.future_head_1(s_prior_flat, sc_flat).view(B, T, -1)
        fut2_logits  = self.future_head_2(s_prior_flat, sc_flat).view(B, T, -1)
        post_logits  = self.posterior_head(s_post_flat, sc_flat).view(B, T, -1)

        # ── Phase 4c: surprise = prediction error magnitude ───────────────────
        surprise = errors.float().norm(dim=-1) * valid_mask.float()      # (B, T)

        return {
            "prediction_logits":    prior_logits,                        # (B, T, E)
            "posterior_logits":     post_logits,                         # (B, T, E)
            "future_logits_1":      fut1_logits,                         # (B, T, E)
            "future_logits_2":      fut2_logits,                         # (B, T, E)
            "dispositional_states": prior_beliefs,                       # (B, T, D)
            "surprise":             surprise,                            # (B, T)
            "sigreg_loss":          self.sigreg(prior_beliefs.view(B * T, -1)),
            "jepa_loss":            jepa_loss,
            "speaker_contexts":     all_spk_ctx,
            "speaker_ids":          speaker_ids,
            "emotion_ids":          emotion_ids,
            "lengths":              lengths,
        }

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ── Staged-training helpers ────────────────────────────────────────────

    def _lora_params(self) -> List[nn.Parameter]:
        """LoRA adapter parameters only (may be empty if use_lora=False)."""
        return [p for name, p in self.llama_encoder.llama.named_parameters()
                if "lora_" in name]

    def _dispositional_params(self) -> List[nn.Parameter]:
        """All parameters except the LLaMA/LoRA encoder."""
        llama_ids = {id(p) for p in self.llama_encoder.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in llama_ids]

    def freeze_lora(self):
        for p in self._lora_params():
            p.requires_grad = False
        print("  LoRA frozen — training dispositional modules only.")

    def unfreeze_lora(self):
        for p in self._lora_params():
            p.requires_grad = True
        print("  LoRA unfrozen — joint training.")

    def rebuild_prediction_head(self, num_emotions: int):
        """Rebuild all prediction heads when switching datasets."""
        scene_dim = self.cfg.scene_state_dim if self.cfg.use_scene_dynamics else 0
        sd        = self.cfg.dispositional_state_dim
        self.prediction_head = PredictionHead(sd, num_emotions, scene_dim)
        self.future_head_1   = PredictionHead(sd, num_emotions, scene_dim)
        self.future_head_2   = PredictionHead(sd, num_emotions, scene_dim)
        self.posterior_head  = PredictionHead(sd, num_emotions, scene_dim)
        self.cfg.num_emotions = num_emotions
