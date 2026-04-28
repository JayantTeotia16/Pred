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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# SIGReg — Gaussian regulariser on dispositional state space (from LeWM)
# ──────────────────────────────────────────────────────────────────────────────

class SIGReg(nn.Module):
    """
    Enforces that the distribution of dispositional states across a batch
    matches an isotropic Gaussian, preventing state-space collapse.

    Method (Cramér-Wold theorem): if all 1D projections of a distribution
    are Gaussian, the full distribution is Gaussian. We project states onto
    M random unit directions and penalise deviation from N(0,1) per projection.

    Loss = mean over projections of [μ² + (σ - 1)²]
    Where μ, σ are mean and std of the projected values.
    """

    def __init__(self, n_projections: int = 256):
        super().__init__()
        self.M = n_projections

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (N, d) — flattened dispositional states from a batch
        d = Z.shape[1]
        u = F.normalize(torch.randn(d, self.M, device=Z.device), dim=0)  # (d, M)
        proj  = Z.float() @ u                          # (N, M)
        mu    = proj.mean(0)                            # (M,)
        sigma = proj.std(0).clamp(min=1e-6)             # (M,)
        return (mu.pow(2) + (sigma - 1.0).pow(2)).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Emotion Label Context  —  per-speaker label-history GRU
# ──────────────────────────────────────────────────────────────────────────────

class EmotionLabelContext(nn.Module):
    """
    Per-speaker GRU that tracks emotion label history — parallel to the text GRU.
    Operates on observed emotion labels (causally valid: labels 0..t-1 are known).
    Complements DynamicSpeakerContext by capturing what the speaker FELT, not just
    what they said.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.context_dim = cfg.label_context_dim
        self.n_emotions  = cfg.num_emotions
        self.embed = nn.Embedding(cfg.num_emotions + 1, cfg.emotion_label_embed_dim)  # +1 = unknown
        self.gru   = nn.GRUCell(cfg.emotion_label_embed_dim, cfg.label_context_dim)
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)

    def init_states(self, B: int, max_speakers: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, max_speakers, self.context_dim, device=device)

    def get_context(self, states: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        B   = states.shape[0]
        idx = speaker_ids.clamp(max=states.shape[1] - 1)
        return states[torch.arange(B, device=states.device), idx]

    def update(self, states: torch.Tensor, speaker_ids: torch.Tensor,
               emotion_ids: torch.Tensor) -> torch.Tensor:
        B, S, _ = states.shape
        idx      = speaker_ids.clamp(max=S - 1)
        safe_emo = torch.where(emotion_ids >= 0, emotion_ids,
                               torch.full_like(emotion_ids, self.n_emotions))
        emb   = self.embed(safe_emo)
        h_old = states[torch.arange(B, device=states.device), idx]
        h_new = self.gru(emb.float(), h_old.float())
        states_new = states.clone()
        states_new[torch.arange(B, device=states.device), idx] = h_new.to(states.dtype)
        return states_new


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Speaker Attention  —  learned upgrade over mean-pooling SRA
# ──────────────────────────────────────────────────────────────────────────────

class CrossSpeakerAttention(nn.Module):
    """
    At turn t, the active speaker attends over OTHER speakers' past δu (turns 0..t-1).
    Replaces the naive causal mean of SRA with a learned multi-head attention,
    letting the model weight which other speakers and which past turns matter most.

    Masking: strictly causal (no attending to t or later) + same-speaker blocked.
    Rows with no valid keys (no other speaker has spoken yet) return zeros.
    """

    def __init__(self, perturbation_dim: int, num_heads: int = 2):
        super().__init__()
        assert perturbation_dim % num_heads == 0
        self.n_heads  = num_heads
        self.head_dim = perturbation_dim // num_heads
        self.scale    = self.head_dim ** -0.5
        self.q_proj   = nn.Linear(perturbation_dim, perturbation_dim, bias=False)
        self.k_proj   = nn.Linear(perturbation_dim, perturbation_dim, bias=False)
        self.v_proj   = nn.Linear(perturbation_dim, perturbation_dim, bias=False)
        self.out_proj = nn.Linear(perturbation_dim, perturbation_dim)

    def forward(
        self,
        delta_u:     torch.Tensor,  # (B, T, pd)
        speaker_ids: torch.Tensor,  # (B, T)
        valid_mask:  torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:              # (B, T, pd)
        B, T, pd = delta_u.shape
        device   = delta_u.device
        H, D     = self.n_heads, self.head_dim

        Q = self.q_proj(delta_u).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.k_proj(delta_u).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(delta_u).view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # Strict causal: query t cannot see key t or later
        causal  = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=0)
        # Same speaker: block self-attention across turns of the same speaker
        same_spk = speaker_ids.unsqueeze(2) == speaker_ids.unsqueeze(1)           # (B, T, T)
        # Invalid keys (padding)
        inv_key  = ~valid_mask.unsqueeze(1).expand(B, T, T)                       # (B, T, T)

        mask = (causal.unsqueeze(0) | same_spk | inv_key).unsqueeze(1)            # (B, 1, T, T)
        mask = mask.expand(B, H, T, T)

        scores = scores.masked_fill(mask, float("-inf"))

        # Rows with no valid keys → zero output (avoid NaN)
        all_masked = mask.all(dim=-1, keepdim=True)
        scores     = scores.masked_fill(all_masked, 0.0)
        weights    = torch.softmax(scores, dim=-1)
        weights    = weights.masked_fill(all_masked, 0.0)

        out = torch.matmul(weights, V)                              # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, pd)
        return self.out_proj(out)


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

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Last-token pooling only — result can be cached across frozen epochs."""
        lengths = mask.sum(1).long() - 1
        lengths = lengths.clamp(min=0)
        B       = hidden.shape[0]
        return hidden[torch.arange(B, device=hidden.device), lengths].float()  # (B, H)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(hidden, mask))


# ──────────────────────────────────────────────────────────────────────────────
# Causal Transformer Dynamics  —  with Speaker-Relational Attention (SRA)
# ──────────────────────────────────────────────────────────────────────────────

class CausalTransformerDynamics(nn.Module):
    """
    Causally-masked transformer over per-turn (δu, c, cross_ctx, emo_embed, label_ctx).

    Upgrades over the original:
      - CrossSpeakerAttention replaces mean-pooling SRA
      - Past emotion label embedding conditions each turn on the active speaker's
        last known emotion (causally valid: labels 0..t-1 are observed)
      - EmotionLabelContext (label GRU) output is concatenated as additional context

    Input per turn: cat(δu_t, spk_ctx_t, cross_ctx_t, emo_embed_t, label_ctx_t)
    Output: dispositional states (B, T, state_dim), strictly causal.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.transformer_dim
        self.n_heads = cfg.transformer_heads

        self.cross_spk_attn = CrossSpeakerAttention(cfg.perturbation_dim, num_heads=2)
        self.emotion_embed  = nn.Embedding(cfg.num_emotions + 1, cfg.emotion_label_embed_dim)

        in_dim = (cfg.perturbation_dim * 2 + cfg.speaker_context_dim
                  + cfg.emotion_label_embed_dim + cfg.label_context_dim)
        self.input_proj = nn.Linear(in_dim, d)
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

        self.initial_state = nn.Parameter(torch.zeros(cfg.dispositional_state_dim))

    def forward(
        self,
        all_delta_u:     torch.Tensor,   # (B, T, perturbation_dim)
        all_speaker_ctx: torch.Tensor,   # (B, T, speaker_context_dim)
        all_label_ctx:   torch.Tensor,   # (B, T, label_context_dim)
        prev_spk_emo:    torch.Tensor,   # (B, T) long — active speaker's last emotion
        valid_mask:      torch.Tensor,   # (B, T) bool
        speaker_ids:     torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        B, T, _ = all_delta_u.shape
        device   = all_delta_u.device

        cross_ctx  = self.cross_spk_attn(all_delta_u, speaker_ids, valid_mask)  # (B, T, pd)
        emo_embeds = self.emotion_embed(prev_spk_emo)                            # (B, T, eld)

        x = torch.cat([all_delta_u, all_speaker_ctx, cross_ctx,
                        emo_embeds, all_label_ctx], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed(torch.arange(T, device=device))

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=~valid_mask)
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

    def set_prior(self, class_counts: torch.Tensor):
        """Initialise output bias to log-prior so the head starts at the right distribution."""
        with torch.no_grad():
            log_prior = torch.log(class_counts.float().clamp(min=1))
            log_prior = log_prior - log_prior.mean()   # centre so softmax ≈ empirical freq
            self.net[-1].bias.copy_(log_prior.to(self.net[-1].bias.device))

    def forward(self, s: torch.Tensor, scene_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = torch.cat([s, scene_s], dim=-1) if scene_s is not None else s
        return self.net(inp)
