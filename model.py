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
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM
from config import ModelConfig
from dispositional_module import (
    PerturbationEncoder,
    DynamicSpeakerContext,
    PersonalDynamicsField,
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
        self.llama = AutoModelForCausalLM.from_pretrained(
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
        self.personal_dynamics  = PersonalDynamicsField(cfg)
        self.scene_dynamics     = SceneDynamicsField(cfg) if cfg.use_scene_dynamics else None
        self.prediction_head    = PredictionHead(
            state_dim       = cfg.dispositional_state_dim,
            num_emotions    = cfg.num_emotions,
            scene_state_dim = cfg.scene_state_dim if cfg.use_scene_dynamics else 0,
        )

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict:
        """
        batch keys:
            input_ids       (B, T, L)
            attention_mask  (B, T, L)
            speaker_ids     (B, T)    — LOCAL ids (0,1,2,...) per conversation
            emotion_ids     (B, T)    — -1 = padding
            length          (B,)
            num_speakers    (B,)      — distinct speakers in each conversation

        Returns:
            prediction_logits    (B, T, num_emotions)
            dispositional_states (B, T, state_dim)
            surprise             (B, T)
            emotion_ids          (B, T)
            lengths              (B,)
        """
        B, T, L  = batch["input_ids"].shape
        device   = batch["input_ids"].device

        input_ids      = batch["input_ids"]        # (B, T, L)
        attention_mask = batch["attention_mask"]   # (B, T, L)
        speaker_ids    = batch["speaker_ids"]      # (B, T)
        emotion_ids    = batch["emotion_ids"]      # (B, T)
        lengths        = batch["length"]           # (B,)

        # ── Pre-encode all utterances in ONE batched LLaMA call ──────────
        # Calling LLaMA T times inside the loop would keep T separate
        # computation graphs alive simultaneously until loss.backward(),
        # causing OOM on 8B models. One batched call (B*T, L) produces a
        # single graph that is freed as soon as backward() runs.
        flat_ids    = input_ids.view(B * T, L)           # (B*T, L)
        flat_mask   = attention_mask.view(B * T, L)      # (B*T, L)
        flat_hidden = self.llama_encoder.encode(flat_ids, flat_mask)       # (B*T, L, H)
        flat_delta  = self.perturbation_enc(flat_hidden, flat_mask)        # (B*T, pd)
        all_delta_u = flat_delta.view(B, T, self.cfg.perturbation_dim)    # (B, T, pd)
        del flat_ids, flat_mask, flat_hidden, flat_delta   # free intermediates early

        # ── Initialise states ─────────────────────────────────────────────

        # Dispositional state — zero for everyone (no prior knowledge)
        s = self.personal_dynamics.initial_state(B, device)       # (B, D)

        # Dynamic speaker context — zero (no utterance history yet)
        spk_ctx = self.speaker_context.init_states(
            B, MAX_LOCAL_SPEAKERS, device
        )                                                          # (B, S, cd)

        # Scene state — zero
        scene_s = None
        if self.scene_dynamics is not None:
            scene_s = self.scene_dynamics.initial_state(B, device) # (B, Ds)

        # δu from the previous turn (initialised to zero)
        prev_delta_u = torch.zeros(B, self.cfg.perturbation_dim, device=device)

        # ── Storage ───────────────────────────────────────────────────────
        all_logits    = []
        all_states    = []
        all_surprise  = []
        all_spk_ctx   = []   # speaker context used at each turn (for contrastive loss)
        prev_logits   = None

        # ── Autoregressive turn loop ──────────────────────────────────────
        for t in range(T):
            spk_t   = speaker_ids[:, t]                   # (B,)
            valid_t = (emotion_ids[:, t] >= 0).float()    # (B,)

            # ── 1. Get current speaker context (built from THEIR past turns)
            c_t = self.speaker_context.get_context(spk_ctx, spk_t)  # (B, cd)
            all_spk_ctx.append(c_t)

            # ── 2. Step ODE: s(t-1) → s(t)
            #       Uses prev_delta_u (from turn t-1) and c_t (speaker context
            #       before seeing turn t) — strictly no leakage from turn t
            s_next, _ = self.personal_dynamics.step(s, c_t, prev_delta_u)

            # Only advance state for non-padding turns
            s = torch.where(valid_t.bool().unsqueeze(-1), s_next, s)

            # ── 3. Predict emotion at turn t (pure prior — zero leakage) ──
            if self.scene_dynamics is not None:
                logits_t = self.prediction_head(s, scene_s)
            else:
                logits_t = self.prediction_head(s)
            all_logits.append(logits_t)
            all_states.append(s)

            # ── 4. Surprise: KL divergence between consecutive predictions ──
            #       Gradients flow so w_surp actually trains the model.
            if prev_logits is not None:
                p_prev = F.softmax(prev_logits, dim=-1)
                p_curr = F.softmax(logits_t, dim=-1)
                surprise_t = F.kl_div(
                    p_prev.log().clamp(min=-10), p_curr, reduction="none"
                ).sum(-1)                                 # (B,)
            else:
                surprise_t = torch.zeros(B, device=device)
            all_surprise.append(surprise_t * valid_t)
            prev_logits = logits_t

            # ── 5. Consume precomputed δu for turn t to update states ─────
            #       Runs AFTER prediction is stored — no leakage.
            #       Gate on valid_t: padding turns must not corrupt states.
            if t < T - 1:
                delta_u = all_delta_u[:, t, :]   # (B, pd) — precomputed above

                # Scene dynamics: update scene, get its influence on δu
                if self.scene_dynamics is not None:
                    scene_s_new, scene_infl = self.scene_dynamics.step(scene_s, s)
                    scene_s = torch.where(
                        valid_t.bool().unsqueeze(-1), scene_s_new, scene_s
                    )
                    delta_u = delta_u + scene_infl

                # Update this speaker's context — only for non-padding turns
                spk_ctx_new = self.speaker_context.update(spk_ctx, spk_t, delta_u)
                spk_ctx = torch.where(
                    valid_t.bool().unsqueeze(-1).unsqueeze(-1), spk_ctx_new, spk_ctx
                )

                # Store δu for next ODE step — only for non-padding turns
                prev_delta_u = torch.where(
                    valid_t.bool().unsqueeze(-1), delta_u, prev_delta_u
                )

        # ── Stack and return ──────────────────────────────────────────────
        return {
            "prediction_logits":    torch.stack(all_logits,   dim=1),  # (B, T, E)
            "dispositional_states": torch.stack(all_states,   dim=1),  # (B, T, D)
            "surprise":             torch.stack(all_surprise, dim=1),  # (B, T)
            "speaker_contexts":     torch.stack(all_spk_ctx,  dim=1),  # (B, T, cd)
            "speaker_ids":          speaker_ids,
            "emotion_ids":          emotion_ids,
            "lengths":              lengths,
        }

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

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
