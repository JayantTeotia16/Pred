"""
prep_data.py — Preprocess HuggingFace datasets into row-per-utterance CSVs.

Usage:
    python prep_data.py --dataset dailydialog --out_dir ./data/dailydialog
    python prep_data.py --dataset iemocap     --out_dir ./data/iemocap
"""

import argparse
import os
import csv


# ── DailyDialog ───────────────────────────────────────────────────────────────
# Emotions: 0=no_emotion→neutral, 1=anger, 2=disgust, 3=fear,
#           4=happiness→joy, 5=sadness, 6=surprise
DAILYDIALOG_INT_TO_NAME = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "joy",     5: "sadness", 6: "surprise",
}


def prep_dailydialog(out_dir: str):
    """
    Uses eusip/silicone dyda_e — DailyDialog already in row-per-utterance format,
    same schema as MELD. Avoids the broken daily_dialog HF dataset zip handling.
    Emotions: 0=no_emotion(neutral),1=anger,2=disgust,3=fear,4=happiness(joy),
              5=sadness,6=surprise
    """
    from datasets import load_dataset
    import json

    print("Downloading DailyDialog via eusip/silicone dyda_e ...")
    split_map = {"train": "train", "val": "validation", "test": "test"}
    os.makedirs(out_dir, exist_ok=True)

    for split_name, hf_split in split_map.items():
        ds = load_dataset("eusip/silicone", "dyda_e",
                          split=hf_split, trust_remote_code=True)
        rows = []
        for sample in ds:
            rows.append({
                "Dialogue_ID":  sample.get("Dialogue_ID", sample.get("dialogue_id", 0)),
                "Utterance_ID": sample.get("Utterance_ID", sample.get("utterance_id", 0)),
                "Utterance":    str(sample.get("Utterance",  sample.get("utterance", ""))).strip(),
                "Speaker":      str(sample.get("Speaker",    sample.get("speaker", "spk_0"))).strip(),
                "Emotion":      sample.get("Emotion", sample.get("emotion", 0)),
            })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)

        total_convs = len(set(str(r["Dialogue_ID"]) for r in rows))
        print(f"  {split_name}: {total_convs} conversations, {len(rows)} utterances → {out_path}")

    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        json.dump(DAILYDIALOG_INT_TO_NAME, f, indent=2)
    print("  DailyDialog prep done.")


# ── IEMOCAP ───────────────────────────────────────────────────────────────────
# Tries the public HuggingFace mirror; falls back with clear instructions.
IEMOCAP_INT_TO_NAME = {
    0: "neutral", 1: "joy", 2: "sadness",
    3: "anger",   4: "fear", 5: "disgust", 6: "surprise",
}


def prep_iemocap(out_dir: str):
    from datasets import load_dataset
    import json

    # Sources to try in order
    SOURCES = [
        ("eusip/silicone", "iemocap"),   # same hub as MELD/DailyDialog, no login needed
        ("Zahra99/iemocap_text", None),
    ]

    ds = None
    for hf_name, hf_cfg in SOURCES:
        try:
            print(f"Attempting IEMOCAP from {hf_name} (config={hf_cfg}) ...")
            ds = load_dataset(hf_name, hf_cfg, trust_remote_code=True)
            print(f"  Success: {hf_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        print("\n[WARN] Could not download IEMOCAP from any source.")
        print("IEMOCAP is a gated dataset. Options:")
        print("  1. Run: huggingface-cli login  then retry")
        print("  2. Download manually and convert to CSV with columns:")
        print("     Dialogue_ID, Utterance_ID, Utterance, Speaker, Emotion")
        print("     where Emotion is one of: neutral,joy,sadness,anger,fear,disgust,surprise")
        print(f"     Place train.csv / val.csv / test.csv in: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Detect column names from the loaded dataset
    sample_split = list(ds.keys())[0]
    cols = list(ds[sample_split].features.keys())
    print(f"  Detected columns: {cols}")

    # Try to auto-map columns
    utt_col  = next((c for c in cols if "text"      in c.lower() or "utterance" in c.lower()), None)
    emo_col  = next((c for c in cols if "emotion"   in c.lower() or "label"     in c.lower()), None)
    spk_col  = next((c for c in cols if "speaker"   in c.lower()), None)
    dial_col = next((c for c in cols if "session"   in c.lower() or "dialogue"  in c.lower()
                                      or "dialog"   in c.lower() or "conv"      in c.lower()), None)
    uid_col  = next((c for c in cols if "utterance_id" in c.lower() or "utt_id" in c.lower()), None)

    print(f"  Mapping: utterance={utt_col}, emotion={emo_col}, "
          f"speaker={spk_col}, dialogue={dial_col}")

    if utt_col is None or emo_col is None:
        print("[ERROR] Could not detect required columns. Please convert manually.")
        return

    split_names = list(ds.keys())
    # Try to map to train/val/test
    def pick(candidates):
        for c in candidates:
            if c in split_names: return c
        return split_names[0] if split_names else None

    split_map = {
        "train": pick(["train"]),
        "val":   pick(["validation", "val", "dev"]),
        "test":  pick(["test"]),
    }
    if split_map["val"] is None:
        split_map["val"] = split_map["train"]

    for split_name, hf_split in split_map.items():
        if hf_split is None:
            continue
        rows = []
        for i, sample in enumerate(ds[hf_split]):
            dial_id = str(sample[dial_col]) if dial_col else str(i)
            utt_id  = str(sample[uid_col])  if uid_col  else str(i)
            speaker = str(sample[spk_col])  if spk_col  else f"spk_{i % 2}"
            utt     = str(sample[utt_col])

            raw_emo = sample[emo_col]
            if isinstance(raw_emo, int):
                emo = raw_emo
            else:
                name = str(raw_emo).strip().lower()
                rev  = {v: k for k, v in IEMOCAP_INT_TO_NAME.items()}
                emo  = rev.get(name, 0)

            rows.append({
                "Dialogue_ID":  dial_id,
                "Utterance_ID": utt_id,
                "Utterance":    utt,
                "Speaker":      speaker,
                "Emotion":      emo,
            })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {split_name}: {len(rows)} utterances → {out_path}")

    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        json.dump(IEMOCAP_INT_TO_NAME, f, indent=2)
    print("  IEMOCAP prep done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True, choices=["dailydialog", "iemocap"])
    p.add_argument("--out_dir",  required=True)
    args = p.parse_args()

    if args.dataset == "dailydialog":
        prep_dailydialog(args.out_dir)
    elif args.dataset == "iemocap":
        prep_iemocap(args.out_dir)
