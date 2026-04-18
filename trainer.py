"""
trainer.py — Loss, training loop, and evaluation.

Dataset-agnostic: works with any emotion label set and any speaker setup.
Speaker names are used only for display in per-character analysis
(optional — falls back gracefully if speaker info is unavailable).
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from config import TrainingConfig
from model import DispositionalPredictionModel


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class PredictionLoss(nn.Module):
    """
    L = w_pred · CE(prediction_logits, labels)
      + w_surp · surprise_calibration_reg

    Supervision starts at min_history_turns to skip cold-start turns
    where the model has no history to work with.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.w_pred   = cfg.prediction_loss_weight
        self.w_surp   = cfg.surprise_reg_weight
        self.min_hist = cfg.min_history_turns
        self.ce       = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        logits   = outputs["prediction_logits"]    # (B, T, E)
        labels   = outputs["emotion_ids"]          # (B, T)
        surprise = outputs["surprise"]             # (B, T)
        B, T, E  = logits.shape

        pred_loss = self.ce(logits.view(B*T, E), labels.view(B*T)).view(B, T)

        # Mask: exclude padding and cold-start turns
        valid = (labels >= 0).float()
        if self.min_hist > 0:
            cold = torch.ones(B, T, device=logits.device)
            cold[:, :self.min_hist] = 0.0
            valid = valid * cold

        # Surprise calibration: penalise high surprise on predictable turns
        surp_reg = surprise * (1.0 - torch.exp(-pred_loss.detach()))

        n       = valid.sum().clamp(min=1)
        L_pred  = (pred_loss * valid).sum() / n
        L_surp  = (surp_reg  * valid).sum() / n
        total   = self.w_pred * L_pred + self.w_surp * L_surp

        return total, {
            "loss_total":      total.item(),
            "loss_prediction": L_pred.item(),
            "loss_surprise":   L_surp.item(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collect predictions (skips padding and cold-start)
# ──────────────────────────────────────────────────────────────────────────────

def collect_predictions(
    outputs: Dict,
    min_history: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns flat (preds, labels, turn_indices) for valid supervised turns.
    """
    logits = outputs["prediction_logits"]   # (B, T, E)
    labels = outputs["emotion_ids"]          # (B, T)
    preds  = logits.argmax(dim=-1)

    mask = (labels >= 0)
    if min_history > 0:
        mask[:, :min_history] = False

    T      = logits.shape[1]
    t_idx  = torch.arange(T, device=logits.device).unsqueeze(0).expand_as(labels)
    return (
        preds[mask].cpu().numpy(),
        labels[mask].cpu().numpy(),
        t_idx[mask].cpu().numpy(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        model: DispositionalPredictionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        emotion_labels: Optional[List[str]] = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.emotion_labels = emotion_labels or [str(i) for i in range(model.cfg.num_emotions)]
        self.device       = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.loss_fn = PredictionLoss(cfg)

        trainable = model.get_trainable_params()
        print(f"  Trainable parameters : {sum(p.numel() for p in trainable):,}")

        self.optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps    = len(train_loader) * cfg.num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.learning_rate,
            total_steps=max(total_steps, 1),
            pct_start=cfg.warmup_steps / max(total_steps, 1),
        )

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.best_val_f1 = 0.0
        self.global_step = 0
        self.history     = []

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ── Training epoch ─────────────────────────────────────────────────────

    def train_epoch(self) -> Dict:
        self.model.train()
        all_preds, all_labels = [], []
        epoch_losses = []

        for batch in self.train_loader:
            batch   = self._to_device(batch)
            outputs = self.model(batch)
            loss, loss_dict = self.loss_fn(outputs)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            epoch_losses.append(loss_dict)
            p, l, _ = collect_predictions(outputs, self.cfg.min_history_turns)
            all_preds.extend(p); all_labels.extend(l)
            self.global_step += 1

            if self.global_step % self.cfg.log_every == 0:
                avg = {k: np.mean([d[k] for d in epoch_losses[-self.cfg.log_every:]])
                       for k in loss_dict}
                print(f"  step {self.global_step} | " +
                      " | ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        wf1 = f1_score(np.array(all_labels), np.array(all_preds),
                       average="weighted", zero_division=0)
        return {"train_wf1": wf1,
                "avg_loss": np.mean([d["loss_total"] for d in epoch_losses])}

    # ── Evaluation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict:
        self.model.eval()
        all_preds, all_labels, all_turns = [], [], []
        all_surprise = []

        for batch in loader:
            batch   = self._to_device(batch)
            outputs = self.model(batch)

            p, l, t = collect_predictions(outputs, self.cfg.min_history_turns)
            all_preds.extend(p); all_labels.extend(l); all_turns.extend(t)

            mask = (outputs["emotion_ids"] >= 0)
            mask[:, :self.cfg.min_history_turns] = False
            all_surprise.extend(outputs["surprise"][mask].cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_turns  = np.array(all_turns)

        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # F1 by history length bucket
        bucket_f1 = {}
        for name, lo, hi in [("1-3", 1, 3), ("4-7", 4, 7), ("8-14", 8, 14), ("15+", 15, 999)]:
            m = (all_turns >= lo) & (all_turns <= hi)
            if m.sum() > 5:
                bucket_f1[name] = float(f1_score(
                    all_labels[m], all_preds[m], average="weighted", zero_division=0
                ))

        report = classification_report(
            all_labels, all_preds,
            target_names=self.emotion_labels[:len(set(all_labels))],
            zero_division=0,
        )

        print(f"\n[{split.upper()}]")
        print(f"  Weighted F1   : {wf1:.4f}")
        print(f"  Mean Surprise : {np.mean(all_surprise):.4f}")
        print(f"  F1 by history : {bucket_f1}")

        return {
            f"{split}_wf1":          wf1,
            f"{split}_mean_surprise": float(np.mean(all_surprise)),
            f"{split}_bucket_f1":     bucket_f1,
            "report":                 report,
        }

    # ── Full run ───────────────────────────────────────────────────────────

    def run(self):
        print("\n=== Training Start ===")
        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.num_epochs}")
            train_m = self.train_epoch()
            val_m   = self.evaluate(self.val_loader, "val")

            wf1 = val_m["val_wf1"]
            if wf1 > self.best_val_f1:
                self.best_val_f1 = wf1
                path = os.path.join(self.cfg.save_dir, "best_model.pt")
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                    "metrics":     {k: v for k, v in val_m.items() if k != "report"},
                    "model_cfg":   self.model.cfg,
                }, path)
                print(f"  *** New best val F1: {wf1:.4f} → saved ***")

            self.history.append({**train_m,
                                  **{k: v for k, v in val_m.items() if k != "report"},
                                  "epoch": epoch})

        print(f"\n=== Done. Best Val F1: {self.best_val_f1:.4f} ===")
        with open(os.path.join(self.cfg.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
