"""
analysis.py — Post-training analysis for the universal dispositional model.

Three analyses:

    1. dispositional_landscapes()
       PCA/t-SNE of dispositional states, coloured by true emotion.
       Optionally split by speaker if speaker info is available.

    2. prediction_by_history_length()
       F1 as a function of how many prior turns the model had.
       Core empirical claim: more history → better dispositional prediction.

    3. seasonal_drift()  [optional — needs season metadata]
       Mean state per (speaker, season) projected to 2D.
       Shows whether the landscape evolves across the dataset's temporal axis.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import ExperimentConfig
from model import DispositionalPredictionModel
from data import ConversationDataset, ID2EMOTION


EMOTION_COLORS = {
    "neutral":  "#95a5a6", "surprise": "#f39c12", "fear":    "#8e44ad",
    "joy":      "#f1c40f", "sadness":  "#2980b9", "disgust": "#27ae60",
    "anger":    "#e74c3c",
}


class DispositionalAnalyser:

    def __init__(
        self,
        model: DispositionalPredictionModel,
        dataset: ConversationDataset,
        cfg: ExperimentConfig,
        output_dir: str = "./analysis_outputs",
    ):
        self.model      = model
        self.dataset    = dataset
        self.cfg        = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device     = next(model.parameters()).device

        # Emotion id → name (from dataset)
        self.id2emotion = {v: k for k, v in dataset.emotion2id.items()}
        self.emotion_colors = {
            name: EMOTION_COLORS.get(name, "#aaaaaa")
            for name in dataset.emotion2id
        }

    # ──────────────────────────────────────────────────────────────────────
    # Collect per-turn records
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_records(self) -> List[Dict]:
        self.model.eval()
        loader  = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=0)
        records = []

        for batch in loader:
            batch   = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            outputs = self.model(batch)

            states   = outputs["dispositional_states"]    # (B, T, D)
            logits   = outputs["prediction_logits"]        # (B, T, E)
            labels   = outputs["emotion_ids"]              # (B, T)
            surprise = outputs["surprise"]                 # (B, T)
            spk_ids  = batch["speaker_ids"]                # (B, T)
            seasons  = batch["season"]                     # (B,)
            lengths  = batch["length"]                     # (B,)
            dial_ids = batch["dialogue_id"]                # list of str

            preds = logits.argmax(dim=-1)

            for b in range(states.shape[0]):
                L      = lengths[b].item()
                season = seasons[b].item() if torch.is_tensor(seasons[b]) else seasons[b]

                for t in range(1, L):    # skip cold-start
                    lbl = labels[b, t].item()
                    if lbl < 0:
                        continue
                    records.append({
                        "dialogue_id": dial_ids[b],
                        "turn_idx":    t,
                        "speaker_id":  spk_ids[b, t].item(),   # local id
                        "season":      season,
                        "state":       states[b, t].cpu().numpy(),
                        "pred":        preds[b, t].item(),
                        "label":       lbl,
                        "surprise":    surprise[b, t].item(),
                    })

        print(f"  Collected {len(records)} valid turn records.")
        return records

    # ──────────────────────────────────────────────────────────────────────
    # 1. Dispositional landscape
    # ──────────────────────────────────────────────────────────────────────

    def dispositional_landscapes(self, records: List[Dict]):
        """
        Projects all dispositional states to 2D, coloured by true emotion.
        If the dataset has speaker info, also produces per-local-speaker plots.
        """
        print("\n[Analysis 1] Dispositional landscape...")

        states = np.array([r["state"] for r in records])
        emos   = [r["label"] for r in records]

        if len(states) < 10:
            print("  Not enough data for landscape plot.")
            return

        reducer   = PCA(n_components=2) if len(states) < 200 else TSNE(
            n_components=2, perplexity=min(30, len(states)//5), random_state=42, n_iter=500
        )
        projected = reducer.fit_transform(states)

        fig, ax = plt.subplots(figsize=(10, 8))
        for emo_id, emo_name in self.id2emotion.items():
            mask = np.array(emos) == emo_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=self.emotion_colors.get(emo_name, "#aaaaaa"),
                label=emo_name, alpha=0.55, s=12, edgecolors="none",
            )
            cx, cy = projected[mask, 0].mean(), projected[mask, 1].mean()
            ax.annotate(emo_name[:3], (cx, cy), fontsize=8,
                        color=self.emotion_colors.get(emo_name, "#333"),
                        fontweight="bold", ha="center")

        ax.set_title("Dispositional State Space\n(colour = true emotion label)",
                     fontsize=13)
        ax.legend(fontsize=8, loc="best", markerscale=2)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "dispositional_landscape.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Prediction F1 by history length
    # ──────────────────────────────────────────────────────────────────────

    def prediction_by_history_length(self, records: List[Dict]) -> Dict:
        """
        F1 broken down by number of prior turns available.
        Rising F1 = dispositional memory accumulating useful information.
        """
        print("\n[Analysis 2] F1 by history length...")

        buckets = [
            ("1-3 turns",  1,  3),
            ("4-7 turns",  4,  7),
            ("8-14 turns", 8, 14),
            ("15+ turns", 15, 999),
        ]

        results = {}
        for name, lo, hi in buckets:
            sub = [r for r in records if lo <= r["turn_idx"] <= hi]
            if not sub:
                continue
            wf1 = f1_score(
                [r["label"] for r in sub], [r["pred"] for r in sub],
                average="weighted", zero_division=0
            )
            results[name] = {"f1": wf1, "n": len(sub)}

        print(f"\n  {'Bucket':<14} {'F1':>8} {'N':>8}")
        print("  " + "-" * 32)
        for name, d in results.items():
            print(f"  {name:<14} {d['f1']:>8.4f} {d['n']:>8}")

        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(results.keys())
        f1s   = [results[n]["f1"] for n in names]
        ns    = [results[n]["n"]  for n in names]
        bars  = ax.bar(names, f1s, color="#3498db", alpha=0.85, edgecolor="white")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"n={n}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, min(1.0, max(f1s) * 1.3 + 0.05))
        ax.set_ylabel("Weighted F1")
        ax.set_title("Prediction F1 vs History Length\n"
                     "(rising = dispositional memory is working)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "prediction_by_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
        return results

    # ──────────────────────────────────────────────────────────────────────
    # 3. Temporal / seasonal drift
    # ──────────────────────────────────────────────────────────────────────

    def temporal_drift(self, records: List[Dict]) -> Dict:
        """
        Mean dispositional state per (dialogue_id bucket) over time.
        Shows whether the model's emotional landscape evolves across
        the corpus's temporal ordering.
        Works without named speakers or season labels.
        """
        print("\n[Analysis 3] Temporal drift...")

        # Group by season proxy
        season_groups: Dict[int, List[np.ndarray]] = defaultdict(list)
        for r in records:
            season_groups[r["season"]].append(r["state"])

        seasons = sorted(season_groups.keys())
        if len(seasons) < 3:
            print("  Not enough temporal variation for drift analysis.")
            return {}

        means = np.array([np.array(season_groups[s]).mean(axis=0) for s in seasons])
        pca   = PCA(n_components=2)
        proj  = pca.fit_transform(means)
        var   = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(proj[:, 0], proj[:, 1], "o-", color="#3498db",
                linewidth=2, markersize=8, zorder=3)
        for i, s in enumerate(seasons):
            ax.annotate(f"S{s}", (proj[i, 0]+0.01, proj[i, 1]+0.01), fontsize=9)
        ax.set_title(
            f"Mean Dispositional State Drift Across Time\n"
            f"(PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%)"
        )
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "temporal_drift.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

        total_drift = float(np.linalg.norm(proj[-1] - proj[0]))
        print(f"  Total drift (first→last): {total_drift:.4f}")
        return {"seasons": seasons, "total_drift": total_drift}

    # ──────────────────────────────────────────────────────────────────────
    # Run all
    # ──────────────────────────────────────────────────────────────────────

    def run_all(self):
        print("\n=== Running Full Analysis Suite ===")
        records = self._collect_records()
        self.dispositional_landscapes(records)
        hist    = self.prediction_by_history_length(records)
        drift   = self.temporal_drift(records)

        summary = {
            "history_f1": {k: v["f1"] for k, v in hist.items()},
            "temporal_drift": drift,
        }
        with open(os.path.join(self.output_dir, "analysis_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Analysis complete → {self.output_dir} ===")
