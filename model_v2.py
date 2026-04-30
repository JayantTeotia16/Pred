"""
model_v2.py — Dispositional model + Reactivity Head for interpretability.

The main model (prior pathway, recognition head) is IDENTICAL to v1.
A separate ReactivityHead is added as an interpretability probe:

    ReactivityHead: s(t) → P(emotion changes at turn t)

Trained with BCE against binary emotion-change labels, on a FROZEN
base model. The prior F1 is completely unaffected — this is purely
an analysis tool.

Why s(t) is the right input:
    s(t) is built from ALL past utterances via CausalTransformerDynamics,
    including cross-speaker attention. It already encodes the influence of
    the other speaker's last utterance implicitly. If s(t) predicts
    transitions better than chance, and if reactive speakers have
    systematically higher reactivity scores, that confirms the
    dispositional state captures emotional contagion.
"""

import torch
import torch.nn as nn

from config import ModelConfig
from model import DispositionalPredictionModel


# ──────────────────────────────────────────────────────────────────────────────
# Reactivity Head
# ──────────────────────────────────────────────────────────────────────────────

class ReactivityHead(nn.Module):
    """
    Predicts P(emotion changes at turn t) from the dispositional state s(t).

    Trained with BCE on emotion-change labels (emotion[t] != emotion[t-1]).
    Output in [0, 1] is the reactivity score:
        high → speaker is about to change emotion (reactive / contagious)
        low  → speaker is maintaining their emotional trajectory (autonomous)

    The head is trained on a frozen base model so it does not affect
    the prior prediction pathway in any way.
    """

    def __init__(self, dispositional_state_dim: int, speaker_context_dim: int):
        super().__init__()
        in_dim = dispositional_state_dim + speaker_context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        disp_states:  torch.Tensor,   # (N, state_dim)
        spk_contexts: torch.Tensor,   # (N, ctx_dim)
    ) -> torch.Tensor:                # (N,) raw logits
        return self.net(torch.cat([disp_states, spk_contexts], dim=-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Extended model  (forward is identical to v1)
# ──────────────────────────────────────────────────────────────────────────────

class DispositionalPredictionModelV2(DispositionalPredictionModel):
    """
    Identical forward pass to DispositionalPredictionModel.
    Adds ReactivityHead as a frozen-model probe trained separately in test_model.py.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.reactivity_head = ReactivityHead(
            cfg.dispositional_state_dim,
            cfg.speaker_context_dim,
        )

    def freeze_base(self):
        """Freeze everything except the reactivity head."""
        for name, p in self.named_parameters():
            p.requires_grad = "reactivity_head" in name
        n_frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen  : {n_frozen:,} params")
        print(f"  Trainable (reactivity head only): {n_trainable:,} params")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
