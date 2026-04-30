"""
test_model.py — ReactivityHead probe on IEMOCAP.

Pipeline
--------
1. Train DispositionalPredictionModelV2 (staged, identical to v1 prior pathway).
2. Freeze base model; train ReactivityHead with BCE on emotion-change labels.
3. Evaluate:
   - AUROC / F1 for transition prediction
   - Mean reactivity at stable vs transition turns
   - Per-speaker reactivity distribution

Results saved to:
  checkpoints_v2/test_metrics.json
  checkpoints_v2/gate_analysis.json

Usage:
  python test_model.py [--device cuda] [--batch_size 4] [--llama_model ...]
                       [--local_data ./data/iemocap]
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from config import ExperimentConfig
from data import build_dataloaders
from model_v2 import DispositionalPredictionModelV2
from trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--local_data",  default="./data/iemocap")
    p.add_argument("--utterance_col",   default="Utterance")
    p.add_argument("--speaker_col",     default="Speaker")
    p.add_argument("--emotion_col",     default="Emotion")
    p.add_argument("--dialogue_id_col", default="Dialogue_ID")
    p.add_argument("--output_dir",  default="./checkpoints_v2")
    p.add_argument("--reactivity_epochs", type=int, default=5)
    p.add_argument("--reactivity_lr",     type=float, default=1e-3)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# BCE training loop for the ReactivityHead
# ──────────────────────────────────────────────────────────────────────────────

def _collect_reactivity_data(model, loader, device, min_history=1):
    """
    One pass over the loader with a frozen base model.
    Returns (disp_states, spk_contexts, labels) as flat tensors.
    """
    model.eval()
    all_disp, all_spk, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="collecting states", unit="batch",
                          dynamic_ncols=True, leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)

            disp  = out["dispositional_states"]   # (B, T, D)
            spk   = out["speaker_contexts"]        # (B, T, cd)
            emos  = out["emotion_ids"]             # (B, T)
            valid = (emos >= 0)                    # (B, T)

            B, T, _ = disp.shape
            for b in range(B):
                for t in range(min_history, T):
                    if not valid[b, t]:
                        continue
                    if t == 0 or not valid[b, t - 1]:
                        continue
                    label = int(emos[b, t].item() != emos[b, t - 1].item())
                    all_disp.append(disp[b, t])
                    all_spk.append(spk[b, t])
                    all_labels.append(label)

    if not all_disp:
        raise RuntimeError("No valid turns found for reactivity training.")

    return (
        torch.stack(all_disp),                                     # (N, D)
        torch.stack(all_spk),                                      # (N, cd)
        torch.tensor(all_labels, dtype=torch.float32).to(device),  # (N,)
    )


def train_reactivity_head(model, train_loader, val_loader, device,
                          num_epochs=5, lr=1e-3, min_history=1):
    """
    Freeze base; train ReactivityHead with BCE.
    Returns final val AUROC.
    """
    model.freeze_base()
    print(f"\n[ReactivityHead] Training for {num_epochs} epochs  lr={lr}")

    optimiser = torch.optim.Adam(model.reactivity_head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Pre-collect states (base is frozen, so this is safe to do once)
    print("  Collecting train states...")
    tr_disp, tr_spk, tr_labels = _collect_reactivity_data(
        model, train_loader, device, min_history)
    print(f"  Train samples: {len(tr_labels)}  "
          f"transition rate: {tr_labels.mean().item():.3f}")

    print("  Collecting val states...")
    va_disp, va_spk, va_labels = _collect_reactivity_data(
        model, val_loader, device, min_history)

    # Dataset / DataLoader for the head only
    tr_ds = torch.utils.data.TensorDataset(tr_disp, tr_spk, tr_labels)
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)

    best_auroc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.reactivity_head.train()
        epoch_loss = 0.0
        for d, s, y in tr_ld:
            logits = model.reactivity_head(d, s)
            loss   = criterion(logits, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(y)

        # Validation
        model.reactivity_head.eval()
        with torch.no_grad():
            val_logits = model.reactivity_head(va_disp, va_spk)
            val_loss   = criterion(val_logits, va_labels).item()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy()
            va_np      = va_labels.cpu().numpy()

        try:
            auroc = roc_auc_score(va_np, val_probs)
        except ValueError:
            auroc = float("nan")

        best_auroc = max(best_auroc, auroc)
        avg_tr = epoch_loss / len(tr_labels)
        print(f"  Epoch {epoch}/{num_epochs}  "
              f"train_loss={avg_tr:.4f}  val_loss={val_loss:.4f}  "
              f"val_AUROC={auroc:.4f}")

    return best_auroc


# ──────────────────────────────────────────────────────────────────────────────
# Test-set analysis
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyse_reactivity(model, loader, device, min_history=1):
    """
    Evaluate ReactivityHead on test set.
    Returns analysis dict (AUROC, per-speaker scores, transition vs stable).
    """
    model.eval()
    model.reactivity_head.eval()

    all_probs, all_labels = [], []
    react_transition, react_stable = [], []
    spk_probs = defaultdict(list)

    for batch in tqdm(loader, desc="[reactivity analysis]", unit="batch",
                      dynamic_ncols=True, leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch)

        disp  = out["dispositional_states"]   # (B, T, D)
        spk   = out["speaker_contexts"]        # (B, T, cd)
        emos  = out["emotion_ids"]             # (B, T)
        spks  = out["speaker_ids"]             # (B, T)
        valid = (emos >= 0)

        B, T, _ = disp.shape
        for b in range(B):
            for t in range(min_history, T):
                if not valid[b, t]:
                    continue
                if t == 0 or not valid[b, t - 1]:
                    continue

                d = disp[b, t].unsqueeze(0)
                s = spk[b, t].unsqueeze(0)
                logit = model.reactivity_head(d, s).item()
                prob  = torch.sigmoid(torch.tensor(logit)).item()
                label = int(emos[b, t].item() != emos[b, t - 1].item())
                spk_id = int(spks[b, t].item())

                all_probs.append(prob)
                all_labels.append(label)
                spk_probs[spk_id].append(prob)

                if label == 1:
                    react_transition.append(prob)
                else:
                    react_stable.append(prob)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    try:
        auroc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auroc = float("nan")

    # F1 at 0.5 threshold
    preds = (all_probs >= 0.5).astype(int)
    f1_trans = float(f1_score(all_labels, preds, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(all_labels, preds, average="macro", zero_division=0))

    result = {
        "auroc":                    auroc,
        "f1_transition_class":      f1_trans,
        "f1_macro":                 f1_macro,
        "mean_reactivity":          float(np.mean(all_probs)),
        "reactivity_at_transition": float(np.mean(react_transition)) if react_transition else None,
        "reactivity_at_stable":     float(np.mean(react_stable))     if react_stable     else None,
        "n_transition_turns":       int(np.sum(all_labels)),
        "n_stable_turns":           int(np.sum(1 - all_labels)),
        "per_speaker_reactivity": {
            str(spk): float(np.mean(vals))
            for spk, vals in sorted(spk_probs.items())
            if vals
        },
    }

    if result["reactivity_at_transition"] and result["reactivity_at_stable"]:
        result["reactivity_lift_at_transition"] = (
            result["reactivity_at_transition"] - result["reactivity_at_stable"]
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    cfg = ExperimentConfig()

    try:
        from transformers import AutoConfig as _AC
        _mc = _AC.from_pretrained(args.llama_model)
        cfg.model.llama_hidden_size = _mc.hidden_size
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size}")
    except Exception:
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size} (from config)")
    cfg.model.llama_model_name = args.llama_model

    cfg.training.device          = args.device
    cfg.training.batch_size      = args.batch_size
    cfg.training.save_dir        = args.output_dir
    cfg.training.staged_training = True

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

    print("\nLoading data...")
    _, _, _, train_loader, val_loader, test_loader = \
        build_dataloaders(cfg.model, cfg.data, cfg.training)
    print(f"  Emotions: {cfg.model.num_emotions} → {cfg.model.emotion_labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Train base model (identical to v1) ───────────────────────────
    print("\nBuilding model V2 ...")
    model = DispositionalPredictionModelV2(cfg.model)
    model.to(device)

    param_total = sum(p.numel() for p in model.parameters())
    param_react = sum(p.numel() for p in model.reactivity_head.parameters())
    print(f"  Total params          : {param_total:,}")
    print(f"  ReactivityHead params : {param_react:,}  ({100*param_react/param_total:.2f}%)")

    print("\n=== Phase 1: Train base model (staged) ===")
    trainer = Trainer(model, train_loader, val_loader, cfg.training,
                      emotion_labels=cfg.model.emotion_labels)
    trainer.run()

    print("\n=== Phase 1 Test Evaluation ===")
    trainer.evaluate(test_loader, "test")

    # ── Phase 2: Freeze base, train ReactivityHead ────────────────────────────
    print("\n=== Phase 2: Train ReactivityHead (frozen base) ===")
    train_reactivity_head(
        model, train_loader, val_loader, device,
        num_epochs=args.reactivity_epochs,
        lr=args.reactivity_lr,
        min_history=cfg.training.min_history_turns,
    )
    model.unfreeze_all()

    # ── Phase 3: Analyse on test set ──────────────────────────────────────────
    print("\n=== Phase 3: Reactivity Analysis (test set) ===")
    react_results = analyse_reactivity(
        model, test_loader, device,
        min_history=cfg.training.min_history_turns,
    )

    print(f"\n  AUROC                      : {react_results['auroc']:.4f}")
    print(f"  F1 (transition class)      : {react_results['f1_transition_class']:.4f}")
    print(f"  F1 (macro)                 : {react_results['f1_macro']:.4f}")
    print(f"  Mean reactivity            : {react_results['mean_reactivity']:.4f}")
    print(f"  Reactivity @ transitions   : {react_results['reactivity_at_transition']:.4f}")
    print(f"  Reactivity @ stable        : {react_results['reactivity_at_stable']:.4f}")
    lift = react_results.get("reactivity_lift_at_transition")
    if lift is not None:
        direction = "reactive ↑ at transitions" if lift > 0 else "no clear pattern"
        print(f"  Reactivity lift            : {lift:.4f}  ({direction})")
    print(f"  Transition turns           : {react_results['n_transition_turns']}")
    print(f"  Stable turns               : {react_results['n_stable_turns']}")
    print(f"\n  Per-speaker reactivity (local IDs):")
    for spk, val in react_results["per_speaker_reactivity"].items():
        bar = "█" * int(val * 20)
        print(f"    Speaker {spk}: {val:.4f}  {bar}")

    gate_path = os.path.join(args.output_dir, "gate_analysis.json")
    with open(gate_path, "w") as f:
        json.dump(react_results, f, indent=2)
    print(f"\n  Reactivity analysis saved → {gate_path}")


if __name__ == "__main__":
    main()
