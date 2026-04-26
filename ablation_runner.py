"""
ablation_runner.py — Runs a single ablation configuration.

Accepts config overrides as CLI args and applies them in-memory —
config.py and all other source files are never modified.

Called by ablation.sh for each experiment.
"""

import argparse
import random
import sys
import os

import numpy as np
import torch

from config import ExperimentConfig
from data import build_dataloaders
from model import DispositionalPredictionModel
from baseline import LLaMAClassifier
from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Single ablation run")

    # Identity
    p.add_argument("--name",        required=True, help="Ablation name (for logging)")
    p.add_argument("--output_dir",  required=True)

    # Hardware
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=4)

    # ── Module ablations ─────────────────────────────────────────────────────
    p.add_argument("--baseline",    action="store_true",
                   help="Use LLaMAClassifier baseline (no dispositional modules)")
    p.add_argument("--no_scene",    action="store_true",
                   help="Disable SceneDynamicsField")
    p.add_argument("--no_lora",     action="store_true",
                   help="Disable LoRA (frozen LLaMA)")
    p.add_argument("--staged_training", action="store_true")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override num_epochs (non-staged runs)")

    # New module dims — set to 1 to effectively disable
    p.add_argument("--label_context_dim",      type=int, default=None)
    p.add_argument("--emotion_label_embed_dim", type=int, default=None)

    # ── Loss weight ablations ─────────────────────────────────────────────────
    p.add_argument("--future_pred_weight_1",   type=float, default=None)
    p.add_argument("--future_pred_weight_2",   type=float, default=None)
    p.add_argument("--posterior_loss_weight",  type=float, default=None)
    p.add_argument("--surprise_reg_weight",    type=float, default=None)
    p.add_argument("--contrastive_loss_weight", type=float, default=None)
    p.add_argument("--focal_gamma",            type=float, default=None)
    p.add_argument("--label_smoothing",        type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Ablation: {args.name}")
    print(f"{'='*60}\n")

    cfg = ExperimentConfig()

    # ── Apply model config overrides ──────────────────────────────────────────
    cfg.model.use_scene_dynamics = not args.no_scene
    cfg.model.use_lora           = not args.no_lora

    if args.label_context_dim is not None:
        cfg.model.label_context_dim = args.label_context_dim
    if args.emotion_label_embed_dim is not None:
        cfg.model.emotion_label_embed_dim = args.emotion_label_embed_dim

    # ── Apply training config overrides ───────────────────────────────────────
    cfg.training.device        = args.device
    cfg.training.batch_size    = args.batch_size
    cfg.training.save_dir      = args.output_dir
    cfg.training.staged_training = args.staged_training and not args.baseline

    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs

    if args.future_pred_weight_1 is not None:
        cfg.training.future_pred_weight_1 = args.future_pred_weight_1
    if args.future_pred_weight_2 is not None:
        cfg.training.future_pred_weight_2 = args.future_pred_weight_2
    if args.posterior_loss_weight is not None:
        cfg.training.posterior_loss_weight = args.posterior_loss_weight
    if args.surprise_reg_weight is not None:
        cfg.training.surprise_reg_weight = args.surprise_reg_weight
    if args.contrastive_loss_weight is not None:
        cfg.training.contrastive_loss_weight = args.contrastive_loss_weight
    if args.focal_gamma is not None:
        cfg.training.focal_gamma = args.focal_gamma
    if args.label_smoothing is not None:
        cfg.training.label_smoothing = args.label_smoothing

    set_seed(cfg.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark       = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print("Loading data...")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        build_dataloaders(cfg.model, cfg.data, cfg.training)
    print(f"  Emotions: {cfg.model.num_emotions} → {cfg.model.emotion_labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.baseline:
        print("  Mode: LLaMA-LoRA baseline")
        model = LLaMAClassifier(cfg.model)
    else:
        model = DispositionalPredictionModel(cfg.model)
    model.to(device)

    trainer = Trainer(model, train_loader, val_loader, cfg.training,
                      emotion_labels=cfg.model.emotion_labels)
    trainer.run()

    print("\n=== Final Test Evaluation ===")
    trainer.evaluate(test_loader, "test")


if __name__ == "__main__":
    main()
