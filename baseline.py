"""
baseline.py — LLaMA-LoRA history-only baseline.

Architecture:
    LLaMA + LoRA → per-utterance pooled representations
    → causal mean-pool over past turns → MLP head → emotion logits

Identical information constraint to DispositionalPredictionModel:
    To predict emotion at turn t, only utterances 0..t-1 are visible.
    The current utterance is NOT seen — same as our model.

What this baseline lacks vs our model:
    - No ODE personal dynamics
    - No GRU speaker context (all speakers share the same aggregation)
    - No scene dynamics
    - History is a flat mean pool, not a structured trajectory

If our model outperforms this, the dispositional modules are adding
genuine value beyond raw LLaMA representations.

Returns the same output dict as DispositionalPredictionModel so
Trainer, PredictionLoss, and evaluate() work completely unchanged.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from transformers import AutoModel
from config import ModelConfig


class LLaMAClassifier(nn.Module):
    """
    LLaMA-LoRA history-only baseline.

    Step 1 — encode all utterances in one batched LLaMA call (B*T, L).
    Step 2 — for each turn t, causal-mean-pool the representations of
             turns 0..t-1 (no current utterance).
    Step 3 — MLP head → emotion logits.

    Turn 0 has no history so its context is zero → masked out by
    min_history_turns=1 in the loss, same as our model.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        print(f"  [Baseline] Loading LLaMA: {cfg.llama_model_name}")
        self.llama = AutoModel.from_pretrained(
            cfg.llama_model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16,
        )

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
            self.llama.enable_input_require_grads()
            self.llama.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in self.llama.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.llama.parameters())
            print(f"  [Baseline] LoRA applied — {trainable:,} / {total:,} trainable params")
        else:
            for p in self.llama.parameters():
                p.requires_grad = False
            print("  [Baseline] LLaMA frozen.")

        self.head = nn.Sequential(
            nn.Linear(cfg.llama_hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, cfg.num_emotions),
        )

    # ── Encoding ───────────────────────────────────────────────────────────

    def _encode_all(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode all (B*T) utterances in one LLaMA call.
        Returns mean-pooled representations: (B, T, H).
        """
        B, T, L = input_ids.shape
        flat_ids  = input_ids.view(B * T, L)
        flat_mask = attention_mask.view(B * T, L)

        if self.cfg.use_lora:
            out = self.llama(input_ids=flat_ids, attention_mask=flat_mask)
        else:
            with torch.no_grad():
                out = self.llama(input_ids=flat_ids, attention_mask=flat_mask)

        hidden = out.hidden_states[-1].float()             # (B*T, L, H)
        mask   = flat_mask.unsqueeze(-1).float()           # (B*T, L, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B*T, H)
        return pooled.view(B, T, -1)                       # (B, T, H)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict:
        B, T, L = batch["input_ids"].shape
        device  = batch["input_ids"].device
        emotion_ids = batch["emotion_ids"]    # (B, T)

        # Encode all utterances: (B, T, H)
        utt_repr = self._encode_all(batch["input_ids"], batch["attention_mask"])

        # Causal mean pool: context for predicting turn t
        # = mean of utterance representations 0..t-1 (no current turn).
        # Mask out padding turns so they don't contribute to the mean.
        valid  = (emotion_ids >= 0).float().unsqueeze(-1)  # (B, T, 1)
        cumsum = torch.cumsum(utt_repr * valid, dim=1)     # (B, T, H)
        counts = torch.cumsum(valid, dim=1)                # (B, T, 1)

        # Shift right by 1: turn t sees accumulated sum of turns 0..t-1
        context        = torch.zeros_like(cumsum)
        context_counts = torch.zeros_like(counts)
        context[:, 1:, :]        = cumsum[:, :-1, :]
        context_counts[:, 1:, :] = counts[:, :-1, :]

        # Mean pool (turn 0 stays zero — no history, masked by loss anyway)
        context = context / context_counts.clamp(min=1e-9)  # (B, T, H)

        logits = self.head(context)                          # (B, T, E)

        return {
            "prediction_logits":    logits,
            "dispositional_states": torch.zeros(B, T, self.cfg.dispositional_state_dim, device=device),
            "surprise":             torch.zeros(B, T, device=device),
            "speaker_contexts":     torch.zeros(B, T, self.cfg.speaker_context_dim, device=device),
            "speaker_ids":          batch["speaker_ids"],
            "emotion_ids":          emotion_ids,
            "lengths":              batch["length"],
        }

    # ── Param helpers (mirrors DispositionalPredictionModel interface) ─────

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _lora_params(self) -> List[nn.Parameter]:
        return [p for name, p in self.llama.named_parameters() if "lora_" in name]

    def _dispositional_params(self) -> List[nn.Parameter]:
        llama_ids = {id(p) for p in self.llama.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in llama_ids]

    def freeze_lora(self):
        for p in self._lora_params():
            p.requires_grad = False
        print("  [Baseline] LoRA frozen.")

    def unfreeze_lora(self):
        for p in self._lora_params():
            p.requires_grad = True
        print("  [Baseline] LoRA unfrozen.")
