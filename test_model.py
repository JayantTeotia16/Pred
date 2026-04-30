"""
test_model.py — Test Counterfactual Speaker Modeling (model_v2) on IEMOCAP.

Trains DispositionalPredictionModelV2 and reports:
  1. Prior F1 / Recognition F1 / gap  (compare to baseline model)
  2. Gate weight analysis:
     - Mean autonomy score per conversation (gate → 1 = autonomous, 0 = reactive)
     - Autonomy at emotion-transition turns vs stable turns
     - Per-speaker autonomy distribution

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
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import ExperimentConfig
from data import build_dataloaders
from model_v2 import DispositionalPredictionModelV2
from trainer import Trainer, collect_predictions


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
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Gate analysis
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyse_gates(model, loader, emotion_labels, device, min_history=1):
    """
    Collect per-turn gate values and correlate with emotion transitions.
    Returns a dict of analysis results.
    """
    model.eval()

    gates_transition = []   # gate values at turns where emotion CHANGES vs prev
    gates_stable     = []   # gate values at turns where emotion stays same
    conv_autonomy    = []   # mean gate per conversation
    spk_gates        = defaultdict(list)  # per local speaker_id

    bar = tqdm(loader, desc="[gate analysis]", unit="batch",
               dynamic_ncols=True, leave=False)

    for batch in bar:
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        outputs = model(batch)

        gates   = outputs["cf_gates"]    # (B, T)
        emos    = outputs["emotion_ids"] # (B, T)
        spks    = outputs["speaker_ids"] # (B, T)
        valid   = (emos >= 0)

        B, T = gates.shape

        for b in range(B):
            valid_turns = [t for t in range(T) if valid[b, t] and t >= min_history]
            if not valid_turns:
                continue

            g_vals  = gates[b, valid_turns].cpu().numpy()
            emo_seq = emos[b].cpu().numpy()
            spk_seq = spks[b].cpu().numpy()

            conv_autonomy.append(float(g_vals.mean()))

            for i, t in enumerate(valid_turns):
                g = float(gates[b, t].cpu())
                spk_gates[int(spk_seq[t])].append(g)

                if t > 0 and valid[b, t-1]:
                    if emo_seq[t] != emo_seq[t-1]:
                        gates_transition.append(g)
                    else:
                        gates_stable.append(g)

    bar.close()

    result = {
        "mean_autonomy":           float(np.mean(conv_autonomy)) if conv_autonomy else None,
        "std_autonomy":            float(np.std(conv_autonomy))  if conv_autonomy else None,
        "gate_at_transition":      float(np.mean(gates_transition)) if gates_transition else None,
        "gate_at_stable":          float(np.mean(gates_stable))     if gates_stable     else None,
        "n_transition_turns":      len(gates_transition),
        "n_stable_turns":          len(gates_stable),
        "per_speaker_autonomy": {
            str(spk): float(np.mean(vals))
            for spk, vals in sorted(spk_gates.items())
            if vals
        },
    }

    if result["gate_at_transition"] and result["gate_at_stable"]:
        result["autonomy_drop_at_transition"] = (
            result["gate_at_stable"] - result["gate_at_transition"]
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    cfg = ExperimentConfig()

    # Model
    try:
        from transformers import AutoConfig as _AC
        _mc = _AC.from_pretrained(args.llama_model)
        cfg.model.llama_hidden_size = _mc.hidden_size
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size}")
    except Exception:
        print(f"  LLaMA hidden size: {cfg.model.llama_hidden_size} (from config)")
    cfg.model.llama_model_name = args.llama_model

    # Training
    cfg.training.device          = args.device
    cfg.training.batch_size      = args.batch_size
    cfg.training.save_dir        = args.output_dir
    cfg.training.staged_training = True

    # Data
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
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        build_dataloaders(cfg.model, cfg.data, cfg.training)
    print(f"  Emotions: {cfg.model.num_emotions} → {cfg.model.emotion_labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nBuilding model V2 (with Counterfactual Speaker Gate)...")
    model = DispositionalPredictionModelV2(cfg.model)
    model.to(device)

    param_total   = sum(p.numel() for p in model.parameters())
    param_cf_gate = sum(p.numel() for p in model.cf_gate.parameters())
    print(f"  Total params   : {param_total:,}")
    print(f"  CF gate params : {param_cf_gate:,}  ({100*param_cf_gate/param_total:.2f}%)")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg.training,
                      emotion_labels=cfg.model.emotion_labels)
    trainer.run()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n=== Final Test Evaluation ===")
    test_metrics = trainer.evaluate(test_loader, "test")

    # ── Gate analysis ─────────────────────────────────────────────────────────
    print("\n=== Counterfactual Gate Analysis ===")
    gate_results = analyse_gates(
        model, test_loader, cfg.model.emotion_labels, device,
        min_history=cfg.training.min_history_turns,
    )

    print(f"\n  Mean autonomy score  : {gate_results['mean_autonomy']:.4f}  "
          f"(±{gate_results['std_autonomy']:.4f})")
    print(f"  Gate @ stable turns  : {gate_results['gate_at_stable']:.4f}")
    print(f"  Gate @ transitions   : {gate_results['gate_at_transition']:.4f}  "
          f"(drop: {gate_results.get('autonomy_drop_at_transition', 0):.4f})")
    print(f"  Transition turns     : {gate_results['n_transition_turns']}")
    print(f"  Stable turns         : {gate_results['n_stable_turns']}")
    print(f"\n  Per-speaker autonomy (local IDs):")
    for spk, val in gate_results["per_speaker_autonomy"].items():
        print(f"    Speaker {spk}: {val:.4f}")

    gate_path = os.path.join(args.output_dir, "gate_analysis.json")
    with open(gate_path, "w") as f:
        json.dump(gate_results, f, indent=2)
    print(f"\n  Gate analysis saved → {gate_path}")


if __name__ == "__main__":
    main()
