"""
dispositional_module.py — Universal dispositional module.

Key change from the MELD-specific version:
    Speaker embeddings (nn.Embedding lookup table) are REMOVED.
    Replaced with a DynamicSpeakerContext module that builds each
    speaker's context vector from their own past utterances via a GRU.

    This means:
        - No fixed vocabulary of known speakers
        - Works for any number of speakers, named or anonymous
        - Works across datasets with no speaker overlap between train/test
        - The ODE is conditioned on OBSERVED behaviour, not assumed identity

Architecture:
    PerturbationEncoder     LLaMA hidden → δu  (utterance affective push)
    DynamicSpeakerContext   per-speaker GRU built from that speaker's
                            past δu vectors → speaker context c ∈ ℝ^D
    ODEFunc                 ds/dt = f(s, c) + gate(s,δu)·g(s, δu, c)
    PersonalDynamicsField   wraps ODEFunc + DynamicSpeakerContext
    SceneDynamicsField      shared scene-level ODE (co-regulation)
    PredictionHead          s(t) → emotion logits (pure prior, no target leakage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Speaker Context  —  replaces nn.Embedding
# ──────────────────────────────────────────────────────────────────────────────

class DynamicSpeakerContext(nn.Module):
    """
    Builds a per-speaker context vector from that speaker's past utterances.

    Each speaker in a conversation has their own GRU hidden state.
    When a speaker takes a turn, their GRU is updated with the
    perturbation vector δu for that utterance.
    Between turns by other speakers, their state is held constant.

    This completely replaces the speaker embedding table:
        Before: e = embedding[speaker_id]       ← fixed, requires known speakers
        After:  c = gru_hidden[speaker_id]      ← built from utterance history

    Speaker IDs are LOCAL to each conversation (0, 1, 2, ...).
    The model never needs a global speaker vocabulary.

    Usage:
        ctx = DynamicSpeakerContext(cfg)
        states = ctx.init_states(B, max_local_speakers=6, device)
        # For each turn t:
        c_t = ctx.get_context(states, speaker_ids_t)   # (B, context_dim)
        states = ctx.update(states, speaker_ids_t, delta_u_t)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.context_dim = cfg.speaker_context_dim
        self.perturbation_dim = cfg.perturbation_dim

        # GRU: input = δu, hidden = speaker context
        self.gru = nn.GRUCell(
            input_size=cfg.perturbation_dim,
            hidden_size=cfg.speaker_context_dim,
        )

        # Project context to ODE conditioning dim
        # (same dim as speaker_context_dim — keeps it simple)
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def init_states(
        self,
        B: int,
        max_local_speakers: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initialise speaker context states to zeros.
        Shape: (B, max_local_speakers, context_dim)
        All speakers start with no history — unknown personality.
        """
        return torch.zeros(B, max_local_speakers, self.context_dim, device=device)

    def get_context(
        self,
        states: torch.Tensor,      # (B, S, context_dim)
        speaker_ids: torch.Tensor, # (B,)  — local speaker index
    ) -> torch.Tensor:
        """
        Retrieve current context vector for the active speaker at each batch item.
        Returns (B, context_dim).
        """
        # Gather the context for the active speaker in each batch item
        idx = speaker_ids.clamp(max=states.shape[1] - 1)   # guard out-of-range
        # (B, context_dim)
        return states[torch.arange(B := states.shape[0], device=states.device), idx]

    def update(
        self,
        states: torch.Tensor,      # (B, S, context_dim)
        speaker_ids: torch.Tensor, # (B,)
        delta_u: torch.Tensor,     # (B, perturbation_dim)
    ) -> torch.Tensor:
        """
        Update the GRU hidden state for the active speaker.
        Other speakers' states are unchanged.
        Returns updated states (B, S, context_dim).
        """
        B, S, D = states.shape
        idx = speaker_ids.clamp(max=S - 1)   # (B,)

        # Get current hidden state for active speaker
        h_old = states[torch.arange(B, device=states.device), idx]   # (B, D)

        # GRU update
        h_new = self.gru(delta_u.float(), h_old.float())             # (B, D)

        # Scatter back — only update the active speaker slot
        states_new = states.clone()
        states_new[torch.arange(B, device=states.device), idx] = h_new

        return states_new


# ──────────────────────────────────────────────────────────────────────────────
# ODE function  ds/dt = f(s, c) + gate(s,δu) · g(s, δu, c)
# ──────────────────────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    Vector field of the personal dynamics ODE.

    c = dynamic speaker context (from DynamicSpeakerContext)
        replaces the old static speaker embedding e.

    f(s, c) : intrinsic dynamics — where does this speaker naturally drift?
    g(s,δu,c): perturbation dynamics — how does a past utterance push the state?
    gate(s,δu): scalar in (0,1) — how reactive is this speaker to utterances?
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        sd  = cfg.dispositional_state_dim
        pd  = cfg.perturbation_dim
        cd  = cfg.speaker_context_dim
        hd  = cfg.ode_hidden_dim

        # Intrinsic dynamics f(s, c)
        self.f_net = nn.Sequential(
            nn.Linear(sd + cd, hd), nn.Tanh(),
            nn.Linear(hd, hd),     nn.Tanh(),
            nn.Linear(hd, sd),
        )

        # Perturbation gate gate(s, δu)
        self.gate = nn.Sequential(
            nn.Linear(sd + pd, 1),
            nn.Sigmoid(),
        )

        # Perturbation dynamics g(s, δu, c)
        self.g_net = nn.Sequential(
            nn.Linear(sd + pd + cd, hd), nn.Tanh(),
            nn.Linear(hd, hd),           nn.Tanh(),
            nn.Linear(hd, sd),
        )

        # Small init — gentle dynamics at the start of training
        for net in [self.f_net, self.g_net, self.gate]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)

        self._c: Optional[torch.Tensor] = None
        self._du: Optional[torch.Tensor] = None

    def set_context(
        self,
        speaker_context: torch.Tensor,                # (B, context_dim)
        perturbation: Optional[torch.Tensor] = None,  # (B, perturbation_dim)
    ):
        self._c  = speaker_context
        self._du = perturbation

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        c  = self._c   # (B, cd)
        du = self._du  # (B, pd) or None

        ds = self.f_net(torch.cat([s, c], dim=-1))

        if du is not None:
            gate_val = self.gate(torch.cat([s, du], dim=-1))
            ds = ds + gate_val * self.g_net(torch.cat([s, du, c], dim=-1))

        return ds


# ──────────────────────────────────────────────────────────────────────────────
# RK4 integrator (self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def rk4_step(func: ODEFunc, s: torch.Tensor, dt: float) -> torch.Tensor:
    t  = torch.zeros(1, device=s.device)
    k1 = func(t, s)
    k2 = func(t, s + 0.5 * dt * k1)
    k3 = func(t, s + 0.5 * dt * k2)
    k4 = func(t, s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation Encoder
# ──────────────────────────────────────────────────────────────────────────────

class PerturbationEncoder(nn.Module):
    """LLaMA hidden states (past utterance) → affective perturbation δu."""

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
        m      = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.proj(pooled)   # (B, perturbation_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Personal Dynamics Field
# ──────────────────────────────────────────────────────────────────────────────

class PersonalDynamicsField(nn.Module):
    """
    Wraps ODEFunc + DynamicSpeakerContext.

    The dispositional state s(t) is advanced by one turn at a time.
    The speaker context c is updated separately (in model.py) so that
    the ODE always sees the context from BEFORE the current utterance.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.state_dim = cfg.dispositional_state_dim
        self.dt        = cfg.ode_dt
        self.ode_func  = ODEFunc(cfg)

    def initial_state(self, B: int, device: torch.device) -> torch.Tensor:
        """
        All speakers start at zero — no prior knowledge.
        The ODE builds dispositional state from utterance history.
        """
        return torch.zeros(B, self.state_dim, device=device)

    def step(
        self,
        s: torch.Tensor,                         # (B, state_dim)
        speaker_context: torch.Tensor,           # (B, context_dim)
        perturbation: Optional[torch.Tensor],    # (B, perturbation_dim) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance dispositional state by one past utterance.
        Returns (s_next, ds_at_current).
        """
        self.ode_func.set_context(speaker_context, perturbation)
        s_next = rk4_step(self.ode_func, s, self.dt)
        t      = torch.zeros(1, device=s.device)
        ds     = self.ode_func(t, s)
        return s_next, ds


# ──────────────────────────────────────────────────────────────────────────────
# Scene Dynamics Field
# ──────────────────────────────────────────────────────────────────────────────

class SceneDynamicsField(nn.Module):
    """
    Shared scene-level ODE capturing the collective emotional atmosphere.
    Coupled to personal dynamics via scene_influence added to δu.
    Completely speaker-agnostic — operates on the aggregate conversation state.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.state_dim = cfg.scene_state_dim

        self.scene_ode = nn.Sequential(
            nn.Linear(cfg.scene_state_dim + cfg.dispositional_state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, cfg.scene_state_dim),
        )
        self.scene_to_perturbation = nn.Linear(cfg.scene_state_dim, cfg.perturbation_dim)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scene_s_next, scene_influence_on_perturbation)."""
        ds_scene   = self.scene_ode(torch.cat([scene_s, speaker_s], dim=-1))
        scene_next = scene_s + dt * ds_scene
        scene_infl = self.scene_to_perturbation(scene_s)   # (B, perturbation_dim)
        return scene_next, scene_infl


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Head
# ──────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """
    s(t) + scene_s → P(emotion at turn t).
    Zero access to turn t's utterance — pure dispositional prior.
    num_emotions is set dynamically from the dataset.
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

    def forward(
        self,
        s: torch.Tensor,
        scene_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inp = torch.cat([s, scene_s], dim=-1) if scene_s is not None else s
        return self.net(inp)   # (B, num_emotions)
