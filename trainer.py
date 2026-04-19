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
from tqdm import tqdm

from config import TrainingConfig
from model import DispositionalPredictionModel


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def contrastive_speaker_loss(
    speaker_contexts: torch.Tensor,  # (B, T, D)
    speaker_ids: torch.Tensor,       # (B, T)
    valid_mask: torch.Tensor,        # (B, T) bool
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss on speaker context vectors.
    Pulls together turns from the same speaker; pushes apart different speakers.
    Operates within each conversation (no cross-conversation negatives).
    """
    B, T, D = speaker_contexts.shape
    device  = speaker_contexts.device
    loss    = torch.tensor(0.0, device=device)
    count   = 0

    for b in range(B):
        mask = valid_mask[b]              # (T,)
        vecs = speaker_contexts[b][mask]  # (N, D)
        spks = speaker_ids[b][mask]       # (N,)
        N    = vecs.shape[0]
        if N < 2:
            continue

        vecs = F.normalize(vecs.float(), dim=-1)
        sim  = vecs @ vecs.T / temperature                         # (N, N)
        eye  = torch.eye(N, dtype=torch.bool, device=device)
        pos  = (spks.unsqueeze(0) == spks.unsqueeze(1)) & ~eye    # same speaker

        if not pos.any():
            continue

        # log-softmax denominator over all non-self pairs
        sim_masked = sim.masked_fill(eye, float("-inf"))
        log_denom  = torch.logsumexp(sim_masked, dim=-1, keepdim=True)  # (N, 1)
        log_prob   = sim - log_denom                                      # (N, N)

        loss  += -(log_prob * pos.float()).sum() / pos.float().sum()
        count += 1

    return loss / max(count, 1)


class PredictionLoss(nn.Module):
    """
    L = w_pred · CE(prediction_logits, labels)
      + w_surp · surprise_calibration_reg
      + w_cont · contrastive_speaker_loss

    Supervision starts at min_history_turns to skip cold-start turns
    where the model has no history to work with.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.w_pred   = cfg.prediction_loss_weight
        self.w_surp   = cfg.surprise_reg_weight
        self.w_cont   = cfg.contrastive_loss_weight
        self.cont_tmp = cfg.contrastive_temperature
        self.min_hist = cfg.min_history_turns
        self.ce       = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        logits   = outputs["prediction_logits"]    # (B, T, E)
        labels   = outputs["emotion_ids"]          # (B, T)
        surprise = outputs["surprise"]             # (B, T)
        spk_ctx  = outputs["speaker_contexts"]     # (B, T, D)
        spk_ids  = outputs.get("speaker_ids")      # (B, T) — may be absent in sanity
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

        n      = valid.sum().clamp(min=1)
        L_pred = (pred_loss * valid).sum() / n
        L_surp = (surp_reg  * valid).sum() / n

        # Contrastive speaker loss (skip if no speaker ids available)
        if spk_ids is not None and self.w_cont > 0:
            L_cont = contrastive_speaker_loss(
                spk_ctx, spk_ids, valid.bool(), self.cont_tmp
            )
        else:
            L_cont = torch.tensor(0.0, device=logits.device)

        total = self.w_pred * L_pred + self.w_surp * L_surp + self.w_cont * L_cont

        return total, {
            "loss_total":       total.item(),
            "loss_prediction":  L_pred.item(),
            "loss_surprise":    L_surp.item(),
            "loss_contrastive": L_cont.item(),
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

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.best_val_f1 = 0.0
        self.global_step = 0
        self.history     = []

        # Build initial optimizer (overridden per-phase in staged training)
        n_epochs = (cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
                    if cfg.staged_training else cfg.num_epochs)
        self._rebuild_optimizer(
            [{"params": model.get_trainable_params(), "lr": cfg.learning_rate}],
            n_epochs,
        )

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _rebuild_optimizer(self, param_groups: List[Dict], n_epochs: int):
        """Build a fresh AdamW + OneCycleLR for a given set of param groups."""
        self.optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        total_steps    = max(len(self.train_loader) * n_epochs, 1)
        pct_start      = min(self.cfg.warmup_steps / total_steps, 0.3)
        max_lrs        = [g["lr"] for g in param_groups]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0],
            total_steps=total_steps,
            pct_start=pct_start,
        )
        n_params = sum(p.numel() for g in param_groups for p in g["params"])
        lr_str   = ", ".join(f"{g['lr']:.1e}" for g in param_groups)
        print(f"  Optimizer: {n_params:,} params | LR(s): [{lr_str}] | {n_epochs} epochs")

    def _save_checkpoint(self, epoch: int, val_m: Dict):
        path = os.path.join(self.cfg.save_dir, "best_model.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "metrics":     {k: v for k, v in val_m.items() if k != "report"},
            "model_cfg":   self.model.cfg,
        }, path)

    # ── Training epoch ─────────────────────────────────────────────────────

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict:
        self.model.train()
        all_preds, all_labels = [], []
        epoch_losses = []

        bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [train]",
                   unit="batch", dynamic_ncols=True, leave=False)

        for batch in bar:
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

            bar.set_postfix(
                loss=f"{loss_dict['loss_total']:.4f}",
                pred=f"{loss_dict['loss_prediction']:.4f}",
                surp=f"{loss_dict['loss_surprise']:.4f}",
                cont=f"{loss_dict['loss_contrastive']:.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

        bar.close()
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

        bar = tqdm(loader, desc=f"[{split.upper()}]", unit="batch",
                   dynamic_ncols=True, leave=False)

        for batch in bar:
            batch   = self._to_device(batch)
            outputs = self.model(batch)

            p, l, t = collect_predictions(outputs, self.cfg.min_history_turns)
            all_preds.extend(p); all_labels.extend(l); all_turns.extend(t)

            mask = (outputs["emotion_ids"] >= 0)
            mask[:, :self.cfg.min_history_turns] = False
            all_surprise.extend(outputs["surprise"][mask].cpu().numpy())

        bar.close()

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

        unique_labels = sorted(set(all_labels))
        target_names  = [
            self.emotion_labels[i] for i in unique_labels
            if i < len(self.emotion_labels)
        ]
        report = classification_report(
            all_labels, all_preds,
            labels=unique_labels,
            target_names=target_names,
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
        if self.cfg.staged_training:
            self._run_staged()
        else:
            self._run_standard()
        with open(os.path.join(self.cfg.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def _run_standard(self):
        total = self.cfg.num_epochs
        epoch_bar = tqdm(range(1, total + 1), desc="Overall progress",
                         unit="epoch", dynamic_ncols=True)
        for epoch in epoch_bar:
            self._do_epoch(epoch, total, epoch_bar, phase_tag="")
        epoch_bar.close()
        print(f"\n=== Done. Best Val F1: {self.best_val_f1:.4f} ===")

    def _run_staged(self):
        cfg = self.cfg
        phases = [
            (1, cfg.phase1_epochs, "P1-warmup",  False),
            (2, cfg.phase2_epochs, "P2-joint",   True),
            (3, cfg.phase3_epochs, "P3-refine",  False),
        ]
        total = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs

        tqdm.write(f"  Staged training: {cfg.phase1_epochs}+{cfg.phase2_epochs}+"
                   f"{cfg.phase3_epochs} = {total} epochs")

        epoch_bar = tqdm(total=total, desc="Overall progress",
                         unit="epoch", dynamic_ncols=True)
        global_epoch = 0

        for phase_num, n_epochs, tag, lora_active in phases:
            if n_epochs == 0:
                continue

            # ── Phase banner ──────────────────────────────────────────────
            labels = {1: "warm-up (LoRA frozen → dispositional only)",
                      2: "joint   (LoRA + dispositional, separate LRs)",
                      3: "refine  (LoRA frozen → dispositional converges)"}
            tqdm.write(f"\n{'─'*55}\n  Phase {phase_num}: {labels[phase_num]}\n{'─'*55}")

            # ── Freeze / unfreeze LoRA ────────────────────────────────────
            if lora_active:
                self.model.unfreeze_lora()
                param_groups = [
                    {"params": self.model._dispositional_params(), "lr": cfg.learning_rate},
                    {"params": self.model._lora_params(),          "lr": cfg.lora_lr},
                ]
            else:
                self.model.freeze_lora()
                param_groups = [
                    {"params": self.model.get_trainable_params(), "lr": cfg.learning_rate},
                ]

            self._rebuild_optimizer(param_groups, n_epochs)

            # ── Epochs in this phase ──────────────────────────────────────
            for _ in range(n_epochs):
                global_epoch += 1
                self._do_epoch(global_epoch, total, epoch_bar, phase_tag=tag)

        epoch_bar.close()
        print(f"\n=== Done. Best Val F1: {self.best_val_f1:.4f} ===")

    def _do_epoch(self, epoch: int, total_epochs: int, epoch_bar, phase_tag: str):
        train_m = self.train_epoch(epoch, total_epochs)
        val_m   = self.evaluate(self.val_loader, "val")
        wf1     = val_m["val_wf1"]

        postfix = dict(val_f1=f"{wf1:.4f}", best=f"{self.best_val_f1:.4f}",
                       train_f1=f"{train_m['train_wf1']:.4f}")
        if phase_tag:
            postfix["phase"] = phase_tag
        epoch_bar.set_postfix(**postfix)
        epoch_bar.update(1)

        if wf1 > self.best_val_f1:
            self.best_val_f1 = wf1
            self._save_checkpoint(epoch, val_m)
            tqdm.write(f"  *** Epoch {epoch}: new best val F1 {wf1:.4f} → saved ***")

        self.history.append({**train_m,
                              **{k: v for k, v in val_m.items() if k != "report"},
                              "epoch": epoch, "phase": phase_tag})
