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
    # LLaMA — past-utterance encoder (always frozen)
    llama_model_name: str  = "meta-llama/Llama-3.2-1B"
    llama_hidden_size: int = 2048
    llama_max_length: int  = 128
    freeze_llama: bool     = True

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

    # Emotion classes — set from DataConfig at runtime
    num_emotions: int = 7
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "surprise", "fear", "sadness", "disgust", "joy", "anger"
    ])

    # ODE step size (one step per past utterance)
    ode_dt: float = 1.0


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
    num_epochs: int      = 25
    learning_rate: float = 3e-4
    weight_decay: float  = 1e-4
    warmup_steps: int    = 100
    grad_clip: float     = 1.0

    prediction_loss_weight: float = 1.0
    surprise_reg_weight: float    = 0.01
    min_history_turns: int        = 1    # turns with no history excluded from loss

    max_conversation_length: int  = 30

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
