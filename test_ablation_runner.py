"""
test_ablation_runner.py — Single-run ablation for new architectural ideas.

Called by test_ablation.sh for each configuration.

Flags
-----
--recognition_loss_weight  float  (default 1.0; set to 0 for Change 1)
--no_staged                       disable staged training
--use_cross_speaker_emo           other-speaker emotion in DynamicSpeakerContext (Change 2)
--use_future_pred                 auxiliary s(t)→emotion[t+1] head (Change 3)
--future_pred_weight       float  (default 0.1)
--use_joint_transition            joint BCE transition head on s(t) (Change 4)
--joint_transition_weight  float  (default 0.1)
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from config import ExperimentConfig, TrainingConfig
from data import build_dataloaders
from model import DispositionalPredictionModel
from model_v3 import DispositionalPredictionModelV3
from trainer import Trainer, PredictionLoss


# ──────────────────────────────────────────────────────────────────────────────
# Extended loss
# ──────────────────────────────────────────────────────────────────────────────

class V3PredictionLoss(PredictionLoss):
    """
    Extends PredictionLoss with:
      future_pred_weight    · CE(s(t) → emotion[t+1])
      joint_transition_weight · BCE(s(t) → transition_label)
    """

    def __init__(self, cfg: TrainingConfig,
                 future_pred_weight: float = 0.1,
                 joint_transition_weight: float = 0.1):
        super().__init__(cfg)
        self.w_future = future_pred_weight
        self.w_trans  = joint_transition_weight
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        total, breakdown = super().forward(outputs)

        if self.w_future > 0 and "future_logits" in outputs:
            fut_logits = outputs["future_logits"].contiguous()   # (B, T-1, E)
            fut_labels = outputs["future_labels"].contiguous()   # (B, T-1)
            fut_valid  = outputs["future_valid"].contiguous()    # (B, T-1) bool
            L_future = self._masked_ce(fut_logits, fut_labels, fut_valid.float())
            total = total + self.w_future * L_future
            breakdown["loss_future"] = L_future.item()

        if self.w_trans > 0 and "transition_logits" in outputs:
            trans_logits = outputs["transition_logits"]  # (B, T-1)
            trans_labels = outputs["transition_labels"]  # (B, T-1) float
            trans_valid  = outputs["transition_valid"]   # (B, T-1) bool
            n_valid = trans_valid.sum()
            if n_valid > 0:
                L_trans = self.bce(trans_logits[trans_valid], trans_labels[trans_valid])
                total   = total + self.w_trans * L_trans
                breakdown["loss_trans"] = L_trans.item()

        breakdown["loss_total"] = total.item()
        return total, breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--name",        required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B")

    # Data
    p.add_argument("--local_data",       default="./data/iemocap")
    p.add_argument("--utterance_col",    default="Utterance")
    p.add_argument("--speaker_col",      default="Speaker")
    p.add_argument("--emotion_col",      default="Emotion")
    p.add_argument("--dialogue_id_col",  default="Dialogue_ID")

    # Change 1
    p.add_argument("--recognition_loss_weight", type=float, default=1.0)
    p.add_argument("--no_staged",  action="store_true")

    # Change 2
    p.add_argument("--use_cross_speaker_emo", action="store_true")

    # Change 3
    p.add_argument("--use_future_pred",   action="store_true")
    p.add_argument("--future_pred_weight", type=float, default=0.1)

    # Change 4
    p.add_argument("--use_joint_transition",    action="store_true")
    p.add_argument("--joint_transition_weight", type=float, default=0.1)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    print(f"\n{'='*60}")
    print(f"  Test Ablation: {args.name}")
    print(f"{'='*60}\n")

    cfg = ExperimentConfig()

    try:
        from transformers import AutoConfig as _AC
        _mc = _AC.from_pretrained(args.llama_model)
        cfg.model.llama_hidden_size = _mc.hidden_size
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size}")
    except Exception:
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size} (from config)")
    cfg.model.llama_model_name = args.llama_model

    cfg.training.device                 = args.device
    cfg.training.batch_size             = args.batch_size
    cfg.training.save_dir               = args.output_dir
    cfg.training.staged_training        = not args.no_staged
    cfg.training.recognition_loss_weight = args.recognition_loss_weight

    cfg.data.hf_dataset_name     = None
    cfg.data.local_data_dir      = args.local_data
    cfg.data.emotion_int_to_name = None
    cfg.data.utterance_col       = args.utterance_col
    cfg.data.speaker_col         = args.speaker_col
    cfg.data.emotion_col         = args.emotion_col
    cfg.data.dialogue_id_col     = args.dialogue_id_col

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    print("Loading data...")
    _, _, _, train_loader, val_loader, test_loader = \
        build_dataloaders(cfg.model, cfg.data, cfg.training)
    print(f"  Emotions: {cfg.model.num_emotions} → {cfg.model.emotion_labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    needs_v3 = (args.use_cross_speaker_emo or
                args.use_future_pred or
                args.use_joint_transition)

    if needs_v3:
        model = DispositionalPredictionModelV3(
            cfg.model,
            use_cross_speaker_emo = args.use_cross_speaker_emo,
            use_future_pred       = args.use_future_pred,
            use_joint_transition  = args.use_joint_transition,
        )
    else:
        model = DispositionalPredictionModel(cfg.model)

    model.to(device)

    param_total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {param_total:,}")
    flags = []
    if args.recognition_loss_weight == 0:        flags.append("no_recog")
    if args.no_staged:                           flags.append("no_staged")
    if args.use_cross_speaker_emo:               flags.append("cross_speaker_emo")
    if args.use_future_pred:                     flags.append(f"future_pred(w={args.future_pred_weight})")
    if args.use_joint_transition:                flags.append(f"joint_trans(w={args.joint_transition_weight})")
    print(f"  Active changes: {', '.join(flags) or 'none (base)'}")

    trainer = Trainer(model, train_loader, val_loader, cfg.training,
                      emotion_labels=cfg.model.emotion_labels)

    # Replace loss if V3 auxiliary losses are active
    if args.use_future_pred or args.use_joint_transition:
        trainer.loss_fn = V3PredictionLoss(
            cfg.training,
            future_pred_weight      = args.future_pred_weight if args.use_future_pred else 0.0,
            joint_transition_weight = args.joint_transition_weight if args.use_joint_transition else 0.0,
        )

    trainer.run()

    print("\n=== Final Test Evaluation ===")
    trainer.evaluate(test_loader, "test")


if __name__ == "__main__":
    main()
