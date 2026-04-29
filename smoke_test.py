"""
smoke_test.py — Quickly validates prep + data loading for all datasets.

Checks:
  1. prep_data.py runs without error (skipped if CSVs already exist)
  2. Dataloaders build without error
  3. Emotion distribution is diverse (not all one class)
  4. One batch loads with correct tensor shapes
  5. No NaN / all-padding batches

No model loading, no training — runs in minutes.

Usage:
    python smoke_test.py [--datasets multidialog emorynlp iemocap dailydialog meld]
    python smoke_test.py --datasets multidialog   # test one only
"""

import argparse
import os
import sys
import subprocess
import traceback
from collections import Counter

import torch

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[96m[INFO]\033[0m"

results = {}


def check(name, cond, msg=""):
    status = PASS if cond else FAIL
    print(f"  {status} {name}" + (f" — {msg}" if msg else ""))
    return cond


def smoke_dataset(key, display, data_dir, run_kwargs, hf_only=False, prep_cmd=None):
    print(f"\n{'='*60}")
    print(f"  {display}")
    print(f"{'='*60}")
    ok = True

    # ── 1. Prep ──────────────────────────────────────────────────────────────
    if hf_only:
        print(f"  {INFO} HuggingFace dataset — no local prep needed")
    else:
        needs_prep = not all(
            os.path.isfile(os.path.join(data_dir, f))
            for f in ["train.csv", "val.csv", "test.csv"]
        )
        if needs_prep:
            # prep_cmd overrides default args (e.g. appraisal needs --source flag)
            if prep_cmd:
                cmd = [sys.executable, "prep_data.py", "--dataset"] + prep_cmd + ["--out_dir", data_dir]
            else:
                cmd = [sys.executable, "prep_data.py", "--dataset", key, "--out_dir", data_dir]
            print(f"  {INFO} Running: {' '.join(cmd)}")
            ret = subprocess.run(cmd, capture_output=False)
            ok &= check("prep_data.py exit code", ret.returncode == 0,
                        f"exit {ret.returncode}")
            if ret.returncode != 0:
                results[key] = "FAIL (prep)"
                return
        else:
            print(f"  {INFO} CSVs already present — skipping prep")

        # Check CSV contents
        import pandas as pd
        for split in ["train", "val", "test"]:
            path = os.path.join(data_dir, f"{split}.csv")
            try:
                df = pd.read_csv(path)
                ok &= check(f"{split}.csv columns",
                            {"Utterance","Emotion","Dialogue_ID"}.issubset(df.columns),
                            f"got {list(df.columns)}")
                emo_counts = Counter(df["Emotion"].tolist())
                n_classes  = len(emo_counts)
                ok &= check(f"{split}.csv emotion diversity",
                            n_classes > 1,
                            f"{n_classes} classes: {dict(emo_counts.most_common(5))}")
            except Exception as e:
                ok &= check(f"{split}.csv readable", False, str(e))

    # ── 2. Dataloader ─────────────────────────────────────────────────────────
    try:
        from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
        from data import build_dataloaders

        cfg = ExperimentConfig()
        cfg.training.batch_size = 2

        # Skip LLaMA download — use a tiny tokenizer-compatible model for smoke test
        cfg.model.llama_model_name = "gpt2"   # fast download, same tokenizer API
        cfg.model.llama_hidden_size = 768      # gpt2 hidden size (not used in data loading)
        cfg.model.llama_max_length  = 32

        for k, v in run_kwargs.items():
            setattr(cfg.data, k, v)

        print(f"  {INFO} Building dataloaders ...")
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
            build_dataloaders(cfg.model, cfg.data, cfg.training)

        ok &= check("num_emotions > 1", cfg.model.num_emotions > 1,
                    f"got {cfg.model.num_emotions} classes: {cfg.model.emotion_labels}")
        ok &= check("train conversations", len(train_ds) > 0,
                    f"{len(train_ds)} conversations")
        ok &= check("val conversations",   len(val_ds)   > 0,
                    f"{len(val_ds)} conversations")
        ok &= check("test conversations",  len(test_ds)  > 0,
                    f"{len(test_ds)} conversations")

        # ── 3. One batch ──────────────────────────────────────────────────────
        batch = next(iter(train_loader))
        ids   = batch["input_ids"]      # (B, T, L)
        emos  = batch["emotion_ids"]    # (B, T)
        B, T, L = ids.shape

        ok &= check("batch input_ids shape", ids.ndim == 3, f"{tuple(ids.shape)}")
        ok &= check("batch no all-padding",
                    (emos >= 0).any().item(), "all emotion_ids are -1")

        valid_emos = emos[emos >= 0].tolist()
        emo_dist   = Counter(valid_emos)
        n_unique   = len(emo_dist)
        ok &= check("batch emotion diversity", n_unique > 1,
                    f"{n_unique} classes in batch: {dict(emo_dist)}")
        ok &= check("no NaN in input_ids",
                    not torch.isnan(ids.float()).any().item())

        print(f"  {INFO} Batch shape: input_ids={tuple(ids.shape)}, "
              f"emotion_ids={tuple(emos.shape)}, "
              f"num_emotions={cfg.model.num_emotions}")

    except Exception:
        ok &= check("dataloader build", False, traceback.format_exc().splitlines()[-1])
        print(traceback.format_exc())

    results[key] = "PASS" if ok else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["multidialog", "emorynlp", "iemocap", "dailydialog", "meld"],
                        choices=["multidialog", "emorynlp", "iemocap", "dailydialog", "meld",
                                 "appraisal_meld", "appraisal_emorynlp",
                                 "appraisal_dailydialog", "appraisal_all"])
    args = parser.parse_args()

    dataset_configs = {
        "multidialog": {
            "display":  "MultiDialog",
            "data_dir": "./data/multidialog",
            "hf_only":  False,
            "kwargs": {
                "hf_dataset_name": None,
                "local_data_dir":  "./data/multidialog",
                "utterance_col":   "Utterance",
                "speaker_col":     "Speaker",
                "emotion_col":     "Emotion",
                "dialogue_id_col": "Dialogue_ID",
                "emotion_int_to_name": None,
            },
        },
        "emorynlp": {
            "display":  "EmoryNLP",
            "data_dir": "./data/emorynlp",
            "hf_only":  False,
            "kwargs": {
                "hf_dataset_name": None,
                "local_data_dir":  "./data/emorynlp",
                "utterance_col":   "Utterance",
                "speaker_col":     "Speaker",
                "emotion_col":     "Emotion",
                "dialogue_id_col": "Dialogue_ID",
                "emotion_int_to_name": None,
            },
        },
        "iemocap": {
            "display":  "IEMOCAP",
            "data_dir": "./data/iemocap",
            "hf_only":  False,
            "kwargs": {
                "hf_dataset_name": None,
                "local_data_dir":  "./data/iemocap",
                "utterance_col":   "Utterance",
                "speaker_col":     "Speaker",
                "emotion_col":     "Emotion",
                "dialogue_id_col": "Dialogue_ID",
                "emotion_int_to_name": None,
            },
        },
        "dailydialog": {
            "display":  "DailyDialog",
            "data_dir": "./data/dailydialog",
            "hf_only":  False,
            "kwargs": {
                "hf_dataset_name": None,
                "local_data_dir":  "./data/dailydialog",
                "utterance_col":   "Utterance",
                "speaker_col":     "Speaker",
                "emotion_col":     "Emotion",
                "dialogue_id_col": "Dialogue_ID",
                "emotion_int_to_name": None,
            },
        },
        "meld": {
            "display":  "MELD",
            "data_dir": None,
            "hf_only":  True,
            "kwargs": {
                "hf_dataset_name": "eusip/silicone",
                "hf_config_name":  "meld_e",
                "emotion_int_to_name": [
                    "neutral", "surprise", "fear", "joy", "sadness", "disgust", "anger"
                ],
            },
        },
        **{
            f"appraisal_{src}": {
                "display":  f"Appraisal-{src.title()}",
                "data_dir": f"./data/appraisal/{src}",
                "hf_only":  False,
                "prep_cmd": ["appraisal", "--source", src],
                "kwargs": {
                    "hf_dataset_name": None,
                    "local_data_dir":  f"./data/appraisal/{src}",
                    "utterance_col":   "Utterance",
                    "speaker_col":     "Speaker",
                    "emotion_col":     "Emotion",
                    "dialogue_id_col": "Dialogue_ID",
                    "emotion_int_to_name": None,
                },
            }
            for src in ["meld", "emorynlp", "dailydialog", "all"]
        },
    }

    for key in args.datasets:
        cfg = dataset_configs[key]
        smoke_dataset(
            key        = key,
            display    = cfg["display"],
            data_dir   = cfg["data_dir"] or "",
            run_kwargs = cfg["kwargs"],
            hf_only    = cfg["hf_only"],
            prep_cmd   = cfg.get("prep_cmd"),
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for key, status in results.items():
        color = "\033[92m" if status == "PASS" else "\033[91m"
        print(f"  {color}{status}\033[0m  {key}")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"\n  {'All datasets OK — ready to train.' if all_pass else 'Fix failures above before training.'}")


if __name__ == "__main__":
    main()
