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
import torch.nn.functional as F
from typing import Dict, List

from transformers import AutoModel
from config import ModelConfig
from dispositional_module import (
    PerturbationEncoder,
    DynamicSpeakerContext,
    CausalTransformerDynamics,
    SceneDynamicsField,
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
        self.personal_dynamics  = CausalTransformerDynamics(cfg)
        self.scene_dynamics     = SceneDynamicsField(cfg) if cfg.use_scene_dynamics else None
        self.prediction_head    = PredictionHead(
            state_dim       = cfg.dispositional_state_dim,
            num_emotions    = cfg.num_emotions,
            scene_state_dim = cfg.scene_state_dim if cfg.use_scene_dynamics else 0,
        )

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict:
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
        flat_ids    = input_ids.view(B * T, L)
        flat_mask   = attention_mask.view(B * T, L)
        flat_hidden = self.llama_encoder.encode(flat_ids, flat_mask)   # (B*T, L, H)
        flat_delta  = self.perturbation_enc(flat_hidden, flat_mask)    # (B*T, pd)
        all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim) # (B, T, pd)
        del flat_ids, flat_mask, flat_hidden, flat_delta

        # ── Phase 2: build speaker contexts turn-by-turn (GRU pass) ──────
        spk_ctx     = self.speaker_context.init_states(B, MAX_LOCAL_SPEAKERS, device)
        all_spk_ctx = []

        for t in range(T):
            spk_t   = speaker_ids[:, t]
            valid_t = valid_mask[:, t].float()
            all_spk_ctx.append(self.speaker_context.get_context(spk_ctx, spk_t))
            if t < T - 1:
                spk_ctx_new = self.speaker_context.update(spk_ctx, spk_t, all_delta_u[:, t])
                spk_ctx = torch.where(
                    valid_t.bool().unsqueeze(-1).unsqueeze(-1), spk_ctx_new, spk_ctx
                )

        all_spk_ctx = torch.stack(all_spk_ctx, dim=1)   # (B, T, cd)

        # ── Phase 3: causal transformer → dispositional states ────────────
        # Returns (B, T, state_dim); state[t] uses only history 0..t-1
        disp_states = self.personal_dynamics(all_delta_u, all_spk_ctx, valid_mask)

        # ── Phase 4: scene dynamics + prediction + surprise ───────────────
        scene_s    = self.scene_dynamics.initial_state(B, device) if self.scene_dynamics else None
        all_logits, all_states, all_surprise = [], [], []
        prev_logits = None

        for t in range(T):
            valid_t = valid_mask[:, t].float()
            s       = disp_states[:, t]

            logits_t = self.prediction_head(s, scene_s)
            all_logits.append(logits_t)
            all_states.append(s)

            if prev_logits is not None:
                p_prev     = F.softmax(prev_logits, dim=-1)
                p_curr     = F.softmax(logits_t,    dim=-1)
                surprise_t = F.kl_div(
                    p_prev.log().clamp(min=-10), p_curr, reduction="none"
                ).sum(-1)
            else:
                surprise_t = torch.zeros(B, device=device)
            all_surprise.append(surprise_t * valid_t)
            prev_logits = logits_t

            if self.scene_dynamics is not None and t < T - 1:
                scene_s_new = self.scene_dynamics.step(scene_s, s)
                scene_s = torch.where(
                    valid_t.bool().unsqueeze(-1), scene_s_new, scene_s
                )

        return {
            "prediction_logits":    torch.stack(all_logits,   dim=1),
            "dispositional_states": torch.stack(all_states,   dim=1),
            "surprise":             torch.stack(all_surprise, dim=1),
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
        """
        Call after loading data if num_emotions changed (e.g. switching datasets).
        Replaces the prediction head with correct output size.
        """
        scene_dim = self.cfg.scene_state_dim if self.cfg.use_scene_dynamics else 0
        self.prediction_head = PredictionHead(
            self.cfg.dispositional_state_dim, num_emotions, scene_dim
        )
        self.cfg.num_emotions = num_emotions
