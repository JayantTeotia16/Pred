"""
dispositional_module.py — Universal dispositional module.

Architecture:
    PerturbationEncoder        LLaMA hidden → δu  (last-token pooling)
    DynamicSpeakerContext      per-speaker GRU → speaker context c ∈ ℝ^D
    CausalTransformerDynamics  causally-masked transformer over
                               (δu, c, cross_ctx) history → s(t)
                               cross_ctx = mean δu from OTHER speakers (SRA)
    FusionGate                 s_prior(t) + δu_t → s_posterior(t)
    SceneDynamicsField         shared scene-level ODE (co-regulation)
    PredictionHead             s(t) [+ scene_s] → emotion logits
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Speaker Context
# ──────────────────────────────────────────────────────────────────────────────

class DynamicSpeakerContext(nn.Module):
    """Per-speaker GRU context built from that speaker's past utterances only."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.context_dim      = cfg.speaker_context_dim
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
        idx      = speaker_ids.clamp(max=S - 1)
        h_old    = states[torch.arange(B, device=states.device), idx]
        h_new    = self.gru(delta_u.float(), h_old.float())
        states_new = states.clone()
        states_new[torch.arange(B, device=states.device), idx] = h_new.to(states.dtype)
        return states_new


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation Encoder  —  last-token pooling
# ──────────────────────────────────────────────────────────────────────────────

class PerturbationEncoder(nn.Module):
    """
    LLaMA hidden states → affective perturbation δu.
    Uses last-token pooling: LLaMA is causal so the last valid token
    has attended to all prior tokens in the utterance.
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
        lengths = mask.sum(1).long() - 1
        lengths = lengths.clamp(min=0)
        B       = hidden.shape[0]
        pooled  = hidden[torch.arange(B, device=hidden.device), lengths]
        return self.proj(pooled.float())


# ──────────────────────────────────────────────────────────────────────────────
# Causal Transformer Dynamics  —  with Speaker-Relational Attention (SRA)
# ──────────────────────────────────────────────────────────────────────────────

def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """ALiBi per-head slopes — geometric sequence as in Press et al. 2022."""
    n = 2 ** math.floor(math.log2(n_heads))
    slopes = torch.pow(2, -torch.arange(1, n + 1) * 8.0 / n)
    if n < n_heads:
        # Handle non-power-of-2 head counts
        extra = torch.pow(2, -torch.arange(1, 2 * (n_heads - n) + 1, 2) * 8.0 / (2 * n))
        slopes = torch.cat([slopes, extra])
    return slopes   # (n_heads,)


def _alibi_bias(T: int, n_heads: int, device: torch.device) -> torch.Tensor:
    """
    Build causal ALiBi bias matrix (n_heads, T, T).
    Entry [h, i, j] = -slope_h * (i - j)  for j <= i,  else -inf (future masked).
    """
    slopes = _alibi_slopes(n_heads).to(device)              # (H,)
    pos    = torch.arange(T, device=device)
    dist   = (pos.unsqueeze(1) - pos.unsqueeze(0)).float()  # (T, T), dist[i,j] = i-j
    bias   = -slopes.view(-1, 1, 1) * dist.unsqueeze(0)     # (H, T, T)
    # mask future positions with -inf
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    bias   = bias.masked_fill(causal.unsqueeze(0), float("-inf"))
    return bias   # (H, T, T)


class CausalTransformerDynamics(nn.Module):
    """
    Causally-masked transformer over per-turn (δu, c, cross_ctx) features.

    Speaker-Relational Attention (SRA):
        cross_ctx[t] = causal mean of δu from speakers OTHER than the current
        speaker at turn t. This gives each turn awareness of how other
        speakers have been behaving — interpersonal emotional dynamics —
        without modifying the attention mechanism itself.

    ALiBi positional encoding (Press et al. 2022):
        Replaces absolute positional embeddings with per-head distance
        penalties on attention scores. Different heads attend at different
        ranges — some focus on recent turns, others on long-range context.
        Fixes the mid-history (4-14 turn) attention dip.

    Input per turn: cat(δu_t, c_t, cross_ctx_t)  →  project to transformer_dim
    Output: dispositional states (B, T, state_dim), strictly causal.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.transformer_dim
        self.n_heads = cfg.transformer_heads
        # Input: δu + speaker_ctx + cross-speaker δu
        self.input_proj = nn.Linear(cfg.perturbation_dim * 2 + cfg.speaker_context_dim, d)
        # No positional embedding table — ALiBi handles position via attention bias

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

        self.initial_state = nn.Parameter(torch.zeros(cfg.dispositional_state_dim))

    def _cross_speaker_ctx(
        self,
        all_delta_u:  torch.Tensor,   # (B, T, pd)
        speaker_ids:  torch.Tensor,   # (B, T)
        valid_mask:   torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:
        """
        For each turn t, compute the causal mean of δu from OTHER speakers
        at positions 0..t-1. Fully vectorised — no Python loop.

        Strategy: exclusive_cumsum(all δu) − exclusive_cumsum(same-speaker δu)
        """
        B, T, pd = all_delta_u.shape
        device   = all_delta_u.device
        valid_f  = valid_mask.float()                         # (B, T)
        weighted = all_delta_u * valid_f.unsqueeze(-1)        # (B, T, pd)

        # Exclusive total cumulative sum: turn t sees sum of 0..t-1
        cum_all     = torch.zeros(B, T, pd, device=device)
        cum_all[:, 1:] = torch.cumsum(weighted, dim=1)[:, :-1]

        cnt_all     = torch.zeros(B, T, device=device)
        cnt_all[:, 1:] = torch.cumsum(valid_f, dim=1)[:, :-1]

        # Per-speaker exclusive cumsum via one-hot scatter
        S        = min(int(speaker_ids.max().item()) + 1, 16)   # cap at MAX_LOCAL_SPEAKERS
        spk_long = speaker_ids.long().clamp(max=S - 1)
        onehot   = torch.zeros(B, T, S, device=device)
        onehot.scatter_(2, spk_long.unsqueeze(-1), 1.0)         # (B, T, S)

        per_spk_w   = weighted.unsqueeze(2) * onehot.unsqueeze(-1)   # (B, T, S, pd)
        per_spk_cum = torch.zeros(B, T, S, pd, device=device)
        per_spk_cum[:, 1:] = torch.cumsum(per_spk_w, dim=1)[:, :-1]

        spk_valid   = onehot * valid_f.unsqueeze(-1)                  # (B, T, S)
        per_spk_cnt = torch.zeros(B, T, S, device=device)
        per_spk_cnt[:, 1:] = torch.cumsum(spk_valid, dim=1)[:, :-1]

        # Gather same-speaker cumulative for each turn
        idx_pd   = spk_long.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, pd)
        same_cum = per_spk_cum.gather(2, idx_pd).squeeze(2)           # (B, T, pd)
        idx_s    = spk_long.unsqueeze(-1)
        same_cnt = per_spk_cnt.gather(2, idx_s).squeeze(-1)           # (B, T)

        cross_cum = cum_all - same_cum                                  # (B, T, pd)
        cross_cnt = (cnt_all - same_cnt).unsqueeze(-1).clamp(min=1e-9) # (B, T, 1)
        return cross_cum / cross_cnt                                    # (B, T, pd)

    def forward(
        self,
        all_delta_u:     torch.Tensor,   # (B, T, perturbation_dim)
        all_speaker_ctx: torch.Tensor,   # (B, T, speaker_context_dim)
        valid_mask:      torch.Tensor,   # (B, T) bool
        speaker_ids:     torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        B, T, _ = all_delta_u.shape
        device   = all_delta_u.device

        cross_ctx = self._cross_speaker_ctx(all_delta_u, speaker_ids, valid_mask)

        x = torch.cat([all_delta_u, all_speaker_ctx, cross_ctx], dim=-1)
        x = self.input_proj(x)

        # ALiBi bias: (H, T, T) — causal masking + distance penalty, no pos embed needed
        alibi = _alibi_bias(T, self.n_heads, device)
        # PyTorch TransformerEncoder expects mask (T, T) or (B*H, T, T);
        # repeat across batch: (B*H, T, T)
        alibi_mask = alibi.unsqueeze(0).expand(
            x.shape[0] * self.n_heads, -1, -1
        ).reshape(x.shape[0] * self.n_heads, T, T)

        out = self.transformer(x, mask=alibi_mask, src_key_padding_mask=~valid_mask,
                               is_causal=False)
        out = self.output_proj(out)   # (B, T, state_dim)

        states        = torch.zeros(B, T, out.shape[-1], device=device)
        states[:, 1:] = out[:, :-1]
        states[:, 0]  = self.initial_state.unsqueeze(0).expand(B, -1)
        return states


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Gate  —  prior + utterance → posterior
# ──────────────────────────────────────────────────────────────────────────────

class FusionGate(nn.Module):
    """
    Combines the dispositional prior s(t) with the current utterance δu_t
    to produce a posterior state s'(t).

        gate    = σ(W · cat(s_prior, δu))
        s_post  = gate ⊙ s_prior + (1−gate) ⊙ proj(δu)

    The gate learns how much to update the prior based on the utterance.
    Interpretable: high gate value = utterance confirms prior expectations;
    low gate value = utterance is a surprise, posterior departs from prior.
    """

    def __init__(self, state_dim: int, perturbation_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_dim + perturbation_dim, state_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(perturbation_dim, state_dim)

    def forward(self, s_prior: torch.Tensor, delta_u: torch.Tensor) -> torch.Tensor:
        gate   = self.gate(torch.cat([s_prior, delta_u], dim=-1))
        return gate * s_prior + (1.0 - gate) * self.proj(delta_u)


# ──────────────────────────────────────────────────────────────────────────────
# Scene Dynamics Field
# ──────────────────────────────────────────────────────────────────────────────

class SceneDynamicsField(nn.Module):
    """Shared scene-level ODE capturing collective emotional atmosphere."""

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

    def step(self, scene_s: torch.Tensor, speaker_s: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        ds = self.scene_ode(torch.cat([scene_s, speaker_s], dim=-1))
        return scene_s + dt * ds


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Head
# ──────────────────────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """s(t) [+ scene_s] → emotion logits. No access to current utterance."""

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
