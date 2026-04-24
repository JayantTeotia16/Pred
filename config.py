"""
config.py — Universal configuration.

The model makes NO assumptions about:
    - How many speakers exist
    - Whether speakers are named or anonymous
    - Whether speakers repeat across conversations
    - What dataset is being used

Speaker personalisation is built DYNAMICALLY from each speaker's
utterance history within the conversation — not from a lookup table.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # LLaMA — past-utterance encoder
    llama_model_name: str  = "meta-llama/Llama-3.1-8B"
    llama_hidden_size: int = 4096
    llama_max_length: int  = 64

    # LoRA fine-tuning on LLaMA (replaces full freeze)
    use_lora: bool      = True
    lora_rank: int      = 8
    lora_alpha: int     = 16
    lora_dropout: float = 0.05

    # Dispositional state dim
    dispositional_state_dim: int = 64

    # Perturbation: LLaMA hidden states → δu
    perturbation_dim: int = 128

    # ODE network hidden dim
    ode_hidden_dim: int = 128

    # ── Dynamic speaker context (replaces fixed speaker embeddings) ────────
    # Each speaker's context is built from their own past utterances only.
    # Starts as zeros; grows richer as the conversation progresses.
    # Works with any number of speakers, named or anonymous.
    speaker_context_dim: int    = 64   # GRU hidden dim per speaker
    speaker_context_layers: int = 1    # GRU depth

    # Scene dynamics (shared affective field across all speakers)
    use_scene_dynamics: bool = True
    scene_state_dim: int     = 32

    # Emotion label conditioning (options 1 & 2)
    emotion_label_embed_dim: int = 16   # embedding dim for past emotion labels
    label_context_dim: int       = 32   # hidden dim of per-speaker emotion-label GRU

    # Emotion classes — set from DataConfig at runtime
    num_emotions: int = 7
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "sadness", "disgust", "joy", "anger"
    ])

    # Causal Transformer Dynamics (replaces ODE PersonalDynamicsField)
    transformer_dim: int         = 128
    transformer_heads: int       = 8
    transformer_layers: int      = 2
    max_conversation_length: int = 64   # positional embedding max length


@dataclass
class DataConfig:
    """
    Dataset-agnostic loading configuration.

    Supports any HuggingFace dataset or local CSV by mapping
    column names and split names to a common internal schema.
    """
    # ── HuggingFace dataset (used if hf_dataset_name is set) ──────────────
    hf_dataset_name: Optional[str] = "eusip/silicone"
    hf_config_name: Optional[str]  = "meld_e"           # dataset config/subset

    # ── Local CSV fallback (used if hf_dataset_name is None) ──────────────
    local_data_dir: Optional[str]  = None   # path containing train/val/test CSVs
    train_file: str  = "train.csv"
    val_file: str    = "val.csv"
    test_file: str   = "test.csv"

    # ── Column name mapping (adapt to any schema) ──────────────────────────
    utterance_col: str            = "Utterance"
    speaker_col: Optional[str]    = "Speaker"       # None → no speaker info
    emotion_col: str              = "Emotion"
    dialogue_id_col: str          = "Dialogue_ID"
    utterance_id_col: Optional[str] = "Utterance_ID"
    season_col: Optional[str]     = None            # None → infer from dial. order

    # ── HuggingFace split names ────────────────────────────────────────────
    train_split: str = "train"
    val_split: str   = "validation"
    test_split: str  = "test"

    # ── Emotion label mapping ──────────────────────────────────────────────
    # Maps dataset integer → canonical emotion name.
    # Set to None to auto-detect from dataset features.
    # For meld_e: [neutral, surprise, fear, joy, sadness, disgust, anger]
    emotion_int_to_name: Optional[List[str]] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "joy", "sadness", "disgust", "anger"
    ])


@dataclass
class TrainingConfig:
    batch_size: int      = 8
    num_epochs: int      = 25   # used only when staged_training=False
    learning_rate: float = 3e-4
    weight_decay: float  = 1e-3  # raised for regularisation
    warmup_steps: int    = 100
    grad_clip: float     = 1.0

    focal_gamma: float             = 0.0   # focal loss: 0 = standard CE, 2 = standard focal
    label_smoothing: float         = 0.0   # label smoothing (0 = disabled)
    prediction_loss_weight: float  = 1.0
    surprise_reg_weight: float     = 0.5
    contrastive_loss_weight: float = 0.01
    sigreg_loss_weight: float      = 0.0    # Gaussian regulariser on dispositional states
    jepa_loss_weight: float        = 0.0    # JEPA: predict next utterance embedding from s(t)
    contrastive_temperature: float = 0.1
    min_history_turns: int         = 1

    # Multi-step future prediction (auxiliary training heads)
    future_pred_weight_1: float = 0.3   # predict e(t+1) from s(t)
    future_pred_weight_2: float = 0.1   # predict e(t+2) from s(t)

    # Prior + posterior fusion
    posterior_loss_weight: float = 0.5  # CE on posterior head during training

    max_conversation_length: int   = 30

    # ── Staged training ───────────────────────────────────────────────────
    staged_training: bool  = False
    phase1_epochs: int     = 8    # warm-up : LoRA frozen, dispositional only
    phase2_epochs: int     = 3    # joint   : LoRA + dispositional co-adapt
    phase3_epochs: int     = 5    # refine  : LoRA frozen, dispositional converges
    phase1_lr: float       = 1e-3 # higher LR for phase 1 (LoRA frozen — safe to push harder)
    lora_lr: float         = 3e-5 # separate (smaller) LR for LoRA in phase 2

    log_every: int = 50
    save_dir: str  = "./checkpoints"
    device: str    = "cuda"


@dataclass
class ExperimentConfig:
    model: ModelConfig       = field(default_factory=ModelConfig)
    data: DataConfig         = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int                = 42
    experiment_name: str     = "dispositional_prediction_universal_v1"
