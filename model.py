"""
model.py — Universal Dispositional Prediction Model.

Forward pass (per turn t):

    BEFORE seeing utterance t:
        c_t      = speaker_context[active_speaker]   ← built from their past turns
        s(t)     = CausalTransformer(δu_{0:t-1}, c_{0:t-1})  ← strictly causal
        logits_t = PredictionHead(s(t))              ← pure prior, zero leakage

    AFTER storing the prediction:
        hidden_t = LLaMA(utterance_t)            ← encode current utterance
        δu_t     = PerturbationEncoder(hidden_t)
        speaker_context = GRU_update(speaker_context, δu_t, active_speaker)
        → ready for next turn

No global speaker vocabulary. All speaker personalisation is
derived from what that speaker has said in THIS conversation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from transformers import AutoModel
from config import ModelConfig
from dispositional_module import (
    PerturbationEncoder,
    DynamicSpeakerContext,
    EmotionLabelContext,
    CausalTransformerDynamics,
    PredictionHead,
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

        self.llama_encoder      = LLaMAEncoder(cfg)
        self.perturbation_enc   = PerturbationEncoder(cfg.llama_hidden_size, cfg.perturbation_dim)
        self.speaker_context    = DynamicSpeakerContext(cfg)
        self.label_context      = EmotionLabelContext(cfg)
        self.personal_dynamics  = CausalTransformerDynamics(cfg)

        self.prediction_head  = PredictionHead(cfg.dispositional_state_dim, cfg.num_emotions)
        self.recognition_head = PredictionHead(cfg.perturbation_dim, cfg.num_emotions)


    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict, delta_u_cache: Optional[Dict] = None) -> Dict:
        """
        Three-phase forward pass:
          Phase 1 — batched LLaMA encode + perturbation (B*T, L)
          Phase 2 — GRU speaker context pass (fast, T steps)
          Phase 3 — causal transformer → dispositional states (batched)

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

        # ── Phase 3: causal transformer → dispositional states ────────────
        disp_states = self.personal_dynamics(
            all_delta_u, all_spk_ctx, all_label_ctx, all_prev_spk_emo, valid_mask, speaker_ids
        )

        # ── Batched predictions ───────────────────────────────────────────────
        s_flat  = disp_states.view(B * T, -1)
        du_flat = all_delta_u.view(B * T, -1)

        prior_logits = self.prediction_head(s_flat).view(B, T, -1)
        recog_logits = self.recognition_head(du_flat.detach()).view(B, T, -1)

        # ── Surprise metric: KL between consecutive prior distributions ───────
        p = F.softmax(prior_logits, dim=-1).clamp(min=1e-9)
        kl = F.kl_div(
            p[:, :-1].log(), p[:, 1:].clamp(min=1e-9), reduction="none"
        ).sum(-1)
        surprise = torch.cat(
            [torch.zeros(B, 1, device=device), kl], dim=1
        ) * valid_mask.float()

        return {
            "prediction_logits":    prior_logits,     # (B, T, E)
            "recognition_logits":   recog_logits,     # (B, T, E)
            "dispositional_states": disp_states,      # (B, T, D)
            "surprise":             surprise,          # (B, T)
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
        """Rebuild prediction heads when switching datasets."""
        self.prediction_head  = PredictionHead(self.cfg.dispositional_state_dim, num_emotions)
        self.recognition_head = PredictionHead(self.cfg.perturbation_dim, num_emotions)
        self.cfg.num_emotions = num_emotions
