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
from sklearn.metrics import f1_score, classification_report, accuracy_score
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
    L = w_pred  · CE(prior_logits,     labels)
      + w_post  · CE(posterior_logits, labels)
      + w_fut1  · CE(future_logits_1,  labels_shifted_+1)
      + w_fut2  · CE(future_logits_2,  labels_shifted_+2)
      + w_surp  · surprise_calibration_reg
      + w_cont  · contrastive_speaker_loss
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.w_pred   = cfg.prediction_loss_weight
        self.w_surp   = cfg.surprise_reg_weight
        self.w_cont   = cfg.contrastive_loss_weight
        self.w_post   = cfg.posterior_loss_weight
        self.w_recog  = cfg.recognition_loss_weight
        self.w_fut1   = cfg.future_pred_weight_1
        self.w_fut2   = cfg.future_pred_weight_2
        self.w_sig    = cfg.sigreg_loss_weight
        self.w_jepa   = cfg.jepa_loss_weight
        self.cont_tmp = cfg.contrastive_temperature
        self.min_hist = cfg.min_history_turns
        self.focal_gamma = cfg.focal_gamma
        self.ce       = nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none",
            label_smoothing=cfg.label_smoothing,
        )

    def _focal_weight(self, ce_loss: torch.Tensor) -> torch.Tensor:
        """Focal weight: (1 - exp(-ce))^gamma. Downweights easy (low-loss) examples."""
        if self.focal_gamma == 0:
            return torch.ones_like(ce_loss)
        return (1.0 - torch.exp(-ce_loss.detach())) ** self.focal_gamma

    def _masked_ce(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, E = logits.shape
        loss = self.ce(logits.view(B * T, E), labels.view(B * T)).view(B, T)
        n    = mask.sum().clamp(min=1)
        return (loss * mask).sum() / n

    def forward(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        logits_prior = outputs["prediction_logits"]          # (B, T, E)
        logits_post  = outputs.get("posterior_logits")       # (B, T, E) or None
        logits_fut1  = outputs.get("future_logits_1")        # (B, T, E) or None
        logits_fut2  = outputs.get("future_logits_2")        # (B, T, E) or None
        labels       = outputs["emotion_ids"]                 # (B, T)
        surprise     = outputs["surprise"]                    # (B, T)
        spk_ctx      = outputs["speaker_contexts"]            # (B, T, D)
        spk_ids      = outputs.get("speaker_ids")
        B, T, E      = logits_prior.shape
        device       = logits_prior.device

        # Standard validity mask (padding + cold-start)
        valid = (labels >= 0).float()
        if self.min_hist > 0:
            cold = torch.ones(B, T, device=device)
            cold[:, :self.min_hist] = 0.0
            valid = valid * cold

        # ── Prior loss (focal + label smoothing) ─────────────────────────
        pred_loss = self.ce(logits_prior.view(B*T, E), labels.view(B*T)).view(B, T)
        focal_w   = self._focal_weight(pred_loss)
        n         = valid.sum().clamp(min=1)
        L_pred    = (pred_loss * focal_w * valid).sum() / n

        # ── Surprise calibration ──────────────────────────────────────────
        surprise = torch.nan_to_num(surprise, nan=0.0)
        surp_reg = surprise * (1.0 - torch.exp(-pred_loss.detach()))
        L_surp   = (surp_reg * valid).sum() / n

        # ── Posterior loss ────────────────────────────────────────────────
        L_post = self._masked_ce(logits_post, labels, valid) \
                 if logits_post is not None and self.w_post > 0 \
                 else torch.tensor(0.0, device=device)

        # ── Recognition loss (utterance-only head, no cold-start mask) ───
        logits_recog = outputs.get("recognition_logits")
        L_recog = self._masked_ce(logits_recog, labels, (labels >= 0).float()) \
                  if logits_recog is not None and self.w_recog > 0 \
                  else torch.tensor(0.0, device=device)

        # ── Future prediction losses (shifted labels, padding-only mask) ──
        pad_valid = (labels >= 0).float()   # no cold-start for auxiliary task

        if logits_fut1 is not None and self.w_fut1 > 0:
            fut1_labels        = torch.full_like(labels, -1)
            fut1_labels[:, :-1] = labels[:, 1:]
            fut1_mask           = pad_valid * (fut1_labels >= 0).float()
            L_fut1 = self._masked_ce(logits_fut1, fut1_labels, fut1_mask)
        else:
            L_fut1 = torch.tensor(0.0, device=device)

        if logits_fut2 is not None and self.w_fut2 > 0:
            fut2_labels        = torch.full_like(labels, -1)
            fut2_labels[:, :-2] = labels[:, 2:]
            fut2_mask           = pad_valid * (fut2_labels >= 0).float()
            L_fut2 = self._masked_ce(logits_fut2, fut2_labels, fut2_mask)
        else:
            L_fut2 = torch.tensor(0.0, device=device)

        # ── Contrastive speaker loss ──────────────────────────────────────
        if spk_ids is not None and self.w_cont > 0:
            L_cont = contrastive_speaker_loss(spk_ctx, spk_ids, valid.bool(), self.cont_tmp)
        else:
            L_cont = torch.tensor(0.0, device=device)

        # ── SIGReg — Gaussian regulariser on dispositional state space ────
        L_sig  = outputs.get("sigreg_loss", torch.tensor(0.0, device=device))

        # ── JEPA — predict next utterance embedding from s(t) ────────────
        L_jepa = outputs.get("jepa_loss",   torch.tensor(0.0, device=device))

        total = (self.w_pred * L_pred + self.w_surp * L_surp +
                 self.w_post * L_post + self.w_recog * L_recog +
                 self.w_fut1 * L_fut1 + self.w_fut2 * L_fut2 +
                 self.w_cont * L_cont + self.w_sig * L_sig + self.w_jepa * L_jepa)

        return total, {
            "loss_total":       total.item(),
            "loss_pred":        L_pred.item(),
            "loss_post":        L_post.item(),
            "loss_recog":       L_recog.item(),
            "loss_fut1":        L_fut1.item(),
            "loss_fut2":        L_fut2.item(),
            "loss_surprise":    L_surp.item(),
            "loss_contrastive": L_cont.item(),
            "loss_sigreg":      L_sig.item(),
            "loss_jepa":        L_jepa.item(),
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
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self._init_head_priors(train_loader)

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

    def _init_head_priors(self, loader: DataLoader):
        """Set prediction head output biases to log class-frequencies from training data."""
        if not hasattr(self.model, "prediction_head"):
            return  # baseline model has no dispositional heads
        counts = torch.zeros(self.model.cfg.num_emotions)
        for batch in loader:
            labels = batch["emotion_ids"]
            for c in range(self.model.cfg.num_emotions):
                counts[c] += (labels == c).sum().item()
        for head in [self.model.prediction_head, self.model.future_head_1,
                     self.model.future_head_2, self.model.posterior_head,
                     self.model.recognition_head]:
            head.set_prior(counts)
        print(f"  Head priors set — counts: {counts.long().tolist()}")

    def _to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    @torch.no_grad()
    def _build_delta_cache(self) -> Dict:
        """
        Precompute delta_u for every conversation with LoRA frozen.
        Stores {dialogue_id: (T, pd) cpu tensor}.
        Called once before each frozen phase; skips 8B LLaMA for all
        subsequent epochs in that phase.
        """
        print("  Building delta_u cache (LoRA frozen — one-time cost)...")
        self.model.eval()
        cache = {}
        for batch in tqdm(self.train_loader, desc="  caching", leave=False,
                          unit="batch", dynamic_ncols=True):
            batch = self._to_device(batch)
            B, T, L = batch["input_ids"].shape
            flat_ids  = batch["input_ids"].view(B * T, L)
            flat_mask = batch["attention_mask"].view(B * T, L)
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                flat_h = self.model.llama_encoder.encode(flat_ids, flat_mask)
                # Cache pooled hidden (before proj MLP) so perturbation_enc stays trainable
                flat_pooled = self.model.perturbation_enc.pool(flat_h, flat_mask)
            pooled = flat_pooled.view(B, T, -1).cpu()   # (B, T, H_llama) on CPU
            for b, did in enumerate(batch["dialogue_id"]):
                cache[did] = pooled[b]
        self.model.train()
        print(f"  Cache built: {len(cache)} conversations.")
        return cache

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

        delta_cache = getattr(self, "_delta_cache", None)

        for batch in bar:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                if delta_cache is not None:
                    outputs = self.model(batch, delta_u_cache=delta_cache)
                else:
                    outputs = self.model(batch)
                loss, loss_dict = self.loss_fn(outputs)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_losses.append(loss_dict)
            p, l, _ = collect_predictions(outputs, self.cfg.min_history_turns)
            all_preds.extend(p); all_labels.extend(l)
            self.global_step += 1

            bar.set_postfix(
                loss=f"{loss_dict['loss_total']:.4f}",
                pred=f"{loss_dict['loss_pred']:.4f}",
                post=f"{loss_dict['loss_post']:.4f}",
                sig=f"{loss_dict['loss_sigreg']:.4f}",
                jepa=f"{loss_dict['loss_jepa']:.4f}",
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
        all_prior, all_post, all_labels, all_turns = [], [], [], []
        all_recog_preds, all_recog_labels = [], []
        all_surprise = []

        bar = tqdm(loader, desc=f"[{split.upper()}]", unit="batch",
                   dynamic_ncols=True, leave=False)

        for batch in bar:
            batch   = self._to_device(batch)
            outputs = self.model(batch)

            p, l, t = collect_predictions(outputs, self.cfg.min_history_turns)
            all_prior.extend(p); all_labels.extend(l); all_turns.extend(t)

            if "posterior_logits" in outputs:
                out_post = {**outputs, "prediction_logits": outputs["posterior_logits"]}
                p_post, _, _ = collect_predictions(out_post, self.cfg.min_history_turns)
                all_post.extend(p_post)

            # Recognition: utterance-only head, no cold-start (every valid turn counts)
            if "recognition_logits" in outputs:
                out_r = {**outputs, "prediction_logits": outputs["recognition_logits"]}
                pr, lr, _ = collect_predictions(out_r, min_history=0)
                all_recog_preds.extend(pr); all_recog_labels.extend(lr)

            mask = (outputs["emotion_ids"] >= 0)
            mask[:, :self.cfg.min_history_turns] = False
            all_surprise.extend(outputs["surprise"][mask].cpu().numpy())

        bar.close()

        all_prior  = np.array(all_prior)
        all_labels = np.array(all_labels)
        all_turns  = np.array(all_turns)

        wf1_prior = f1_score(all_labels, all_prior, average="weighted", zero_division=0)
        wf1_post  = f1_score(all_labels, np.array(all_post), average="weighted", zero_division=0) \
                    if all_post else None
        wf1_recog = f1_score(np.array(all_recog_labels), np.array(all_recog_preds),
                             average="weighted", zero_division=0) \
                    if all_recog_preds else None

        # F1 by history length bucket
        bucket_f1 = {}
        for name, lo, hi in [("1-3", 1, 3), ("4-7", 4, 7), ("8-14", 8, 14), ("15+", 15, 999)]:
            m = (all_turns >= lo) & (all_turns <= hi)
            if m.sum() > 5:
                bucket_f1[name] = float(f1_score(
                    all_labels[m], all_prior[m], average="weighted", zero_division=0
                ))

        unique_labels = sorted(set(all_labels))
        target_names  = [self.emotion_labels[i] for i in unique_labels
                         if i < len(self.emotion_labels)]

        acc          = accuracy_score(all_labels, all_prior)
        report_dict  = classification_report(all_labels, all_prior,
                                             labels=unique_labels,
                                             target_names=target_names,
                                             output_dict=True,
                                             zero_division=0)
        report_text  = classification_report(all_labels, all_prior,
                                             labels=unique_labels,
                                             target_names=target_names,
                                             zero_division=0)

        print(f"\n[{split.upper()}]")
        print(f"  Prior F1      : {wf1_prior:.4f}")
        if wf1_recog is not None:
            gap = wf1_recog - wf1_prior
            print(f"  Recognition F1: {wf1_recog:.4f}  (gap vs prior {gap:+.4f})")
        if wf1_post is not None:
            gap = wf1_post - wf1_prior
            print(f"  Posterior F1  : {wf1_post:.4f}  (gap vs prior {gap:+.4f})")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  Mean Surprise : {np.mean(all_surprise):.4f}")
        print(f"  F1 by history : {bucket_f1}")
        print(f"\n  Per-emotion (prior):")
        print(f"  {'Emotion':<12} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Support':>9}")
        print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
        for name in target_names:
            m = report_dict[name]
            print(f"  {name:<12} {m['f1-score']:>7.4f} {m['precision']:>7.4f}"
                  f" {m['recall']:>7.4f} {int(m['support']):>9}")

        per_emotion = {name: report_dict[name] for name in target_names}

        if split == "test":
            metrics_out = {
                "weighted_f1":     wf1_prior,
                "recognition_f1":  wf1_recog,
                "posterior_f1":    wf1_post,
                "accuracy":        acc,
                "mean_surprise":   float(np.mean(all_surprise)),
                "bucket_f1":       bucket_f1,
                "per_emotion":     per_emotion,
            }
            out_path = os.path.join(self.cfg.save_dir, "test_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"\n  Metrics saved → {out_path}")

        return {
            f"{split}_wf1":           wf1_prior,
            f"{split}_recog_wf1":     wf1_recog,
            f"{split}_post_wf1":      wf1_post,
            f"{split}_accuracy":      acc,
            f"{split}_mean_surprise": float(np.mean(all_surprise)),
            f"{split}_bucket_f1":     bucket_f1,
            f"{split}_per_emotion":   per_emotion,
            "report":                 report_text,
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
        self._delta_cache = None
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
                self._delta_cache = None   # LoRA updating — cache invalid
                param_groups = [
                    {"params": self.model._dispositional_params(), "lr": cfg.learning_rate},
                    {"params": self.model._lora_params(),          "lr": cfg.lora_lr},
                ]
            else:
                self.model.freeze_lora()
                self._delta_cache = self._build_delta_cache()  # skip 8B for this phase
                phase_lr = cfg.phase1_lr if phase_num == 1 else cfg.learning_rate
                param_groups = [
                    {"params": self.model.get_trainable_params(), "lr": phase_lr},
                ]

            self._rebuild_optimizer(param_groups, n_epochs)

            # ── Epochs in this phase ──────────────────────────────────────
            do_val = (phase_num == 3)   # only validate in phase 3
            for _ in range(n_epochs):
                global_epoch += 1
                self._do_epoch(global_epoch, total, epoch_bar, phase_tag=tag, do_val=do_val)

        epoch_bar.close()
        print(f"\n=== Done. Best Val F1: {self.best_val_f1:.4f} ===")

    def _do_epoch(self, epoch: int, total_epochs: int, epoch_bar, phase_tag: str,
                  do_val: bool = True):
        train_m = self.train_epoch(epoch, total_epochs)

        postfix = dict(train_f1=f"{train_m['train_wf1']:.4f}")
        if phase_tag:
            postfix["phase"] = phase_tag

        if do_val:
            val_m = self.evaluate(self.val_loader, "val")
            wf1   = val_m["val_wf1"]
            postfix.update(val_f1=f"{wf1:.4f}", best=f"{self.best_val_f1:.4f}")
        else:
            val_m = {"val_wf1": None}
            wf1   = None

        epoch_bar.set_postfix(**postfix)
        epoch_bar.update(1)

        if wf1 is not None and wf1 > self.best_val_f1:
            self.best_val_f1 = wf1
            self._save_checkpoint(epoch, val_m)
            tqdm.write(f"  *** Epoch {epoch}: new best val F1 {wf1:.4f} → saved ***")

        self.history.append({**train_m,
                              **{k: v for k, v in val_m.items() if k != "report"},
                              "epoch": epoch, "phase": phase_tag})
