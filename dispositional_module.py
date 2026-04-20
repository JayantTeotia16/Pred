"""
dispositional_module.py — Universal dispositional module.

Architecture:
    PerturbationEncoder        LLaMA hidden → δu  (last-token pooling)
    DynamicSpeakerContext      per-speaker GRU built from that speaker's
                               past δu vectors → speaker context c ∈ ℝ^D
    CausalTransformerDynamics  causally-masked transformer over (δu, c) history
                               → dispositional state s(t) at each turn
    SceneDynamicsField         shared scene-level ODE (co-regulation)
    PredictionHead             s(t) → emotion logits (pure prior, no target leakage)

Why transformer over ODE:
    The ODE was a Markov recurrence — it saw only the immediately previous
    δu and lost salient past events through the bottleneck. The causal
    transformer attends directly to any past turn, eliminating:
      - The mid-history dip (no integration drift)
      - The single prev_delta_u bottleneck
      - ODE numerical instability in long conversations
"""

import torch
import torch.nn as nn
from typing import Optional

from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Speaker Context  —  GRU per speaker, built from utterance history
# ──────────────────────────────────────────────────────────────────────────────

class DynamicSpeakerContext(nn.Module):
    """
    Builds a per-speaker context vector from that speaker's past utterances.

    Each speaker in a conversation has their own GRU hidden state.
    When a speaker takes a turn, their GRU is updated with δu for that
    utterance. Between turns by other speakers, their state is held constant.

    Speaker IDs are LOCAL to each conversation (0, 1, 2, ...).
    The model never needs a global speaker vocabulary.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.context_dim     = cfg.speaker_context_dim
        self.perturbation_dim = cfg.perturbation_dim

        self.gru = nn.GRUCell(
            input_size=cfg.perturbation_dim,
            hidden_size=cfg.speaker_context_dim,
        )
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def init_states(self, B: int, max_local_speakers: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, max_local_speakers, self.context_dim, device=device)

    def get_context(self, states: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        B   = states.shape[0]
        idx = speaker_ids.clamp(max=states.shape[1] - 1)
        return states[torch.arange(B, device=states.device), idx]

    def update(self, states: torch.Tensor, speaker_ids: torch.Tensor, delta_u: torch.Tensor) -> torch.Tensor:
        B, S, _ = states.shape
        idx   = speaker_ids.clamp(max=S - 1)
        h_old = states[torch.arange(B, device=states.device), idx]
        h_new = self.gru(delta_u.float(), h_old.float())
        states_new = states.clone()
        states_new[torch.arange(B, device=states.device), idx] = h_new
        return states_new


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation Encoder  —  LLaMA hidden → δu  (last-token pooling)
# ──────────────────────────────────────────────────────────────────────────────

class PerturbationEncoder(nn.Module):
    """
    LLaMA hidden states (past utterance) → affective perturbation δu.

    Uses last-token pooling: LLaMA is a causal model so the last valid
    token has already attended to all previous tokens in the utterance.
    Last-token pooling captures this richer representation vs mean pooling.
    """

    def __init__(self, llama_hidden_size: int, perturbation_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llama_hidden_size, perturbation_dim * 2),
            nn.GELU(),
            nn.Linear(perturbation_dim * 2, perturbation_dim),
            nn.LayerNorm(perturbation_dim),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, H)   mask: (B, L)
        # Use the last valid token — richest representation in a causal model
        lengths = mask.sum(1).long() - 1          # (B,) index of last valid token
        lengths = lengths.clamp(min=0)
        B       = hidden.shape[0]
        pooled  = hidden[torch.arange(B, device=hidden.device), lengths]  # (B, H)
        return self.proj(pooled.float())


# ──────────────────────────────────────────────────────────────────────────────
# Causal Transformer Dynamics  —  replaces PersonalDynamicsField (ODE)
# ──────────────────────────────────────────────────────────────────────────────

class CausalTransformerDynamics(nn.Module):
    """
    Causally-masked transformer over the sequence of (δu_t, c_t) pairs.

    State at turn t is computed from inputs at turns 0..t-1 only (no leakage).

    Implementation:
        1. Project cat(δu, c) to transformer dim, add positional embedding.
        2. Apply transformer with causal mask (position t attends to 0..t).
        3. Shift output right by 1: state[t] = transformer_out[t-1].
        4. state[0] = learned initial parameter (no history available).

    Advantages over ODE:
        - Attends directly to any past turn — no Markov bottleneck
        - No integration drift for long conversations (fixes 15+ bucket)
        - Multi-head attention learns different emotional timescales
        - LayerNorm on output stabilises state magnitude
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.transformer_dim

        self.input_proj = nn.Linear(cfg.perturbation_dim + cfg.speaker_context_dim, d)
        self.pos_embed  = nn.Embedding(cfg.max_conversation_length, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.transformer_heads,
            dim_feedforward=d * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d, cfg.dispositional_state_dim),
            nn.LayerNorm(cfg.dispositional_state_dim),
        )

        # Learned initial state for turn 0 — no history, start from zero
        self.initial_state = nn.Parameter(torch.zeros(cfg.dispositional_state_dim))

    def forward(
        self,
        all_delta_u:     torch.Tensor,  # (B, T, perturbation_dim)
        all_speaker_ctx: torch.Tensor,  # (B, T, speaker_context_dim)
        valid_mask:      torch.Tensor,  # (B, T) bool  — True = valid turn
    ) -> torch.Tensor:
        """Returns dispositional states (B, T, state_dim), strictly causal."""
        B, T, _ = all_delta_u.shape
        device   = all_delta_u.device

        x = torch.cat([all_delta_u, all_speaker_ctx], dim=-1)  # (B, T, pd+cd)
        x = self.input_proj(x)                                  # (B, T, d)

        pos = torch.arange(T, device=device)
        x   = x + self.pos_embed(pos)                           # (B, T, d)

        # Causal mask: position t can attend to 0..t (upper triangle = -inf)
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )

        # key_padding_mask: True = ignore this key position (padding turns)
        out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=~valid_mask,
        )
        out = self.output_proj(out)   # (B, T, state_dim)

        # Shift right: state[t] uses transformer output from t-1
        states        = torch.zeros(B, T, out.shape[-1], device=device)
        states[:, 1:] = out[:, :-1]
        states[:, 0]  = self.initial_state.unsqueeze(0).expand(B, -1)

        return states   # (B, T, state_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Scene Dynamics Field
# ──────────────────────────────────────────────────────────────────────────────

class SceneDynamicsField(nn.Module):
    """
    Shared scene-level ODE capturing the collective emotional atmosphere.
    Coupled to personal dynamics via the prediction head (scene state
    is concatenated to dispositional state before emotion logits).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.state_dim = cfg.scene_state_dim

        self.scene_ode = nn.Sequential(
            nn.Linear(cfg.scene_state_dim + cfg.dispositional_state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, cfg.scene_state_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def initial_state(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.state_dim, device=device)

    def step(
        self,
        scene_s: torch.Tensor,    # (B, scene_state_dim)
        speaker_s: torch.Tensor,  # (B, dispositional_state_dim)
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Returns scene_s_next."""
        ds_scene = self.scene_ode(torch.cat([scene_s, speaker_s], dim=-1))
        return scene_s + dt * ds_scene


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Head
# ──────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """
    s(t) + scene_s → P(emotion at turn t).
    Zero access to turn t's utterance — pure dispositional prior.
    """

    def __init__(self, state_dim: int, num_emotions: int, scene_state_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + scene_state_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_emotions),
        )

    def forward(self, s: torch.Tensor, scene_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = torch.cat([s, scene_s], dim=-1) if scene_s is not None else s
        return self.net(inp)
