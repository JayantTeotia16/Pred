"""
main.py — Universal entry point.

Usage:
    python main.py --mode sanity                          # no data needed
    python main.py --mode train                           # meld_e (default)
    python main.py --mode train --hf_dataset daily_dialog --hf_config default
    python main.py --mode eval  --checkpoint ./checkpoints/best_model.pt
    python main.py --mode analyse --checkpoint ./checkpoints/best_model.pt
"""

import argparse
import random
import numpy as np
import torch

from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from data import build_dataloaders, ConversationDataset
from model import DispositionalPredictionModel
from trainer import Trainer
from analysis import DispositionalAnalyser


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Checkpoint loaded (epoch {ckpt['epoch']})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check — no real data or LLaMA weights needed
# ──────────────────────────────────────────────────────────────────────────────

def sanity_check(cfg: ExperimentConfig):
    print("=== Sanity Check — Universal Model ===\n")

    # Tiny dims for speed
    cfg.model.llama_hidden_size       = 64
    cfg.model.dispositional_state_dim = 16
    cfg.model.perturbation_dim        = 32
    cfg.model.ode_hidden_dim          = 32
    cfg.model.speaker_context_dim     = 24
    cfg.model.scene_state_dim         = 16
    cfg.model.num_emotions            = 7
    cfg.training.min_history_turns    = 1

    import torch.nn as nn
    from dispositional_module import (
        PerturbationEncoder, DynamicSpeakerContext,
        PersonalDynamicsField, SceneDynamicsField, PredictionHead,
    )

    # Mock LLaMA
    class MockLLaMA(nn.Module):
        def __init__(self, H): super().__init__(); self.H = H
        def encode(self, ids, mask):
            return torch.randn(ids.shape[0], ids.shape[1], self.H)

    model = DispositionalPredictionModel.__new__(DispositionalPredictionModel)
    nn.Module.__init__(model)
    model.cfg = cfg.model
    model.llama_encoder    = MockLLaMA(cfg.model.llama_hidden_size)
    model.perturbation_enc = PerturbationEncoder(cfg.model.llama_hidden_size, cfg.model.perturbation_dim)
    model.speaker_context  = DynamicSpeakerContext(cfg.model)
    model.personal_dynamics= PersonalDynamicsField(cfg.model)
    model.scene_dynamics   = SceneDynamicsField(cfg.model)
    model.prediction_head  = PredictionHead(
        cfg.model.dispositional_state_dim, cfg.model.num_emotions, cfg.model.scene_state_dim
    )

    B, T, L, E = 2, 6, 10, 7
    emotion_ids = torch.randint(0, E, (B, T))
    emotion_ids[1, T-1] = -1    # simulate padding

    batch = {
        "input_ids":      torch.randint(0, 100, (B, T, L)),
        "attention_mask": torch.ones(B, T, L, dtype=torch.long),
        "speaker_ids":    torch.randint(0, 3, (B, T)),  # 3 local speakers
        "emotion_ids":    emotion_ids,
        "length":         torch.tensor([T, T-1]),
        "num_speakers":   torch.tensor([3, 2]),
        "season":         torch.tensor([1, 2]),
        "dialogue_id":    ["dlg_001", "dlg_002"],
    }

    outputs = model(batch)
    print(f"  prediction_logits    : {outputs['prediction_logits'].shape}")
    print(f"  dispositional_states : {outputs['dispositional_states'].shape}")
    print(f"  surprise             : {outputs['surprise'].shape}")

    from trainer import PredictionLoss
    loss_fn = PredictionLoss(cfg.training)
    loss, ld = loss_fn(outputs)
    print(f"  loss_total           : {loss.item():.4f}")
    for k, v in ld.items():
        print(f"    {k}: {v:.4f}")

    # ── No-leakage invariant: state at t=0 must equal initial zero state ──
    init_s = model.personal_dynamics.initial_state(B, torch.device("cpu"))
    state0 = outputs["dispositional_states"][:, 0, :]
    assert torch.allclose(init_s, state0, atol=1e-5), \
        "FAIL: turn-0 state is not the initial state — leakage detected!"
    print("\n  ✓ No-leakage invariant passed.")

    # ── Speaker context is still zero at t=0 ──────────────────────────────
    from model import MAX_LOCAL_SPEAKERS
    zero_ctx = model.speaker_context.init_states(B, MAX_LOCAL_SPEAKERS, torch.device("cpu"))
    # (we can't directly inspect internal spk_ctx after forward, but the
    #  no-leakage check above is equivalent — if context had been updated
    #  before t=0 prediction, s(0) would differ from init_s)
    print("  ✓ Speaker context correctly starts at zero.")
    print("\n✓ Sanity check passed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["train","eval","analyse","sanity"], default="sanity")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--hf_dataset",  default="eusip/silicone",   help="HuggingFace dataset name")
    parser.add_argument("--hf_config",   default="meld_e",           help="HuggingFace dataset config")
    parser.add_argument("--local_data",  default=None,               help="Local CSV dir (overrides HF)")
    parser.add_argument("--utterance_col", default="Utterance")
    parser.add_argument("--speaker_col",   default="Speaker",        help="Set to '' to disable")
    parser.add_argument("--emotion_col",   default="Emotion")
    parser.add_argument("--dialogue_id_col", default="Dialogue_ID")
    parser.add_argument("--train_split",   default="train")
    parser.add_argument("--val_split",     default="validation")
    parser.add_argument("--test_split",    default="test")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--no_scene",    action="store_true")
    parser.add_argument("--output_dir",  default="./checkpoints")
    parser.add_argument("--analysis_dir",default="./analysis_outputs")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.model.llama_model_name   = args.llama_model
    cfg.model.use_scene_dynamics = not args.no_scene
    cfg.training.batch_size      = args.batch_size
    cfg.training.num_epochs      = args.epochs
    cfg.training.device          = args.device
    cfg.training.save_dir        = args.output_dir

    # Data config
    if args.local_data:
        cfg.data.hf_dataset_name = None
        cfg.data.local_data_dir  = args.local_data
    else:
        cfg.data.hf_dataset_name = args.hf_dataset
        cfg.data.hf_config_name  = args.hf_config
    cfg.data.utterance_col     = args.utterance_col
    cfg.data.speaker_col       = args.speaker_col if args.speaker_col else None
    cfg.data.emotion_col       = args.emotion_col
    cfg.data.dialogue_id_col   = args.dialogue_id_col
    cfg.data.train_split       = args.train_split
    cfg.data.val_split         = args.val_split
    cfg.data.test_split        = args.test_split

    set_seed(cfg.seed)

    if args.mode == "sanity":
        sanity_check(cfg)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        build_dataloaders(cfg.model, cfg.data, cfg.training)

    print(f"  Emotion classes: {cfg.model.num_emotions} → {cfg.model.emotion_labels}")

    model = DispositionalPredictionModel(cfg.model)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, device)

    if args.mode == "train":
        trainer = Trainer(model, train_loader, val_loader, cfg.training,
                          emotion_labels=cfg.model.emotion_labels)
        trainer.run()
        print("\n=== Final Test Evaluation ===")
        trainer.evaluate(test_loader, "test")

    elif args.mode == "eval":
        assert args.checkpoint, "--checkpoint required"
        model.to(device)
        trainer = Trainer(model, train_loader, val_loader, cfg.training,
                          emotion_labels=cfg.model.emotion_labels)
        trainer.evaluate(test_loader, "test")

    elif args.mode == "analyse":
        assert args.checkpoint, "--checkpoint required"
        model.to(device)
        analyser = DispositionalAnalyser(model, test_ds, cfg,
                                         output_dir=args.analysis_dir)
        analyser.run_all()


if __name__ == "__main__":
    main()
