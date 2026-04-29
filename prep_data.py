"""
prep_data.py — Preprocess HuggingFace datasets into row-per-utterance CSVs.

Emotion labels are stored as original dataset strings — no mapping to a
canonical cross-dataset vocabulary. Each dataset keeps its own label space.

Usage:
    python prep_data.py --dataset dailydialog --out_dir ./data/dailydialog
    python prep_data.py --dataset iemocap     --out_dir ./data/iemocap
"""

import argparse
import os
import csv


# ── DailyDialog ───────────────────────────────────────────────────────────────
# Original labels (silicone dyda_e returns strings; fallback int map if needed)
DAILYDIALOG_INT_TO_NAME = {
    0: "no_emotion", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness",  5: "sadness", 6: "surprise",
}
# Normalize string variants returned by the HF dataset
DAILYDIALOG_STR_NORM = {
    "no emotion": "no_emotion",
    "no_emotion": "no_emotion",
    "happiness":  "happiness",
    "happy":      "happiness",
}


def prep_dailydialog(out_dir: str):
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
            raw = sample.get("Emotion", sample.get("emotion", 0))
            try:
                emo_name = DAILYDIALOG_INT_TO_NAME.get(int(raw), "no_emotion")
            except (ValueError, TypeError):
                raw_str = str(raw).strip().lower()
                emo_name = DAILYDIALOG_STR_NORM.get(raw_str, raw_str.replace(" ", "_"))
            rows.append({
                "Dialogue_ID":  sample.get("Dialogue_ID", sample.get("dialogue_id", 0)),
                "Utterance_ID": sample.get("Utterance_ID", sample.get("utterance_id", 0)),
                "Utterance":    str(sample.get("Utterance",  sample.get("utterance", ""))).strip(),
                "Speaker":      str(sample.get("Speaker",    sample.get("speaker", "spk_0"))).strip(),
                "Emotion":      emo_name,
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
# Original IEMOCAP short-code → full name mapping (no remapping to other datasets)
IEMOCAP_EMO_MAP = {
    "neu": "neutral",
    "ang": "anger",
    "sad": "sadness",
    "hap": "happiness",
    "exc": "excitement",
    "fru": "frustration",
    "sur": "surprise",
    "fea": "fear",
    "dis": "disgust",
}


def _parse_iemocap_txt(path: str):
    """
    Parse one IEMOCAP txt file (Berzerker/IEMOCAP format).
    Each line: ID<TAB>emotion<TAB>utterance
    ID format: Ses01F_impro01_F000
      → dialogue_id = Ses01F_impro01
      → speaker     = F or M
      → utt_id      = 000
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            utt_id_full, raw_emo, utterance = parts[0], parts[1].strip(), parts[2].strip()
            segments = utt_id_full.rsplit("_", 1)
            if len(segments) != 2:
                continue
            dial_id = segments[0]
            spk_utt = segments[1]
            speaker = spk_utt[0]
            utt_num = spk_utt[1:]

            emo_name = IEMOCAP_EMO_MAP.get(raw_emo.lower(), raw_emo.lower())

            rows.append({
                "Dialogue_ID":  dial_id,
                "Utterance_ID": utt_num,
                "Utterance":    utterance,
                "Speaker":      speaker,
                "Emotion":      emo_name,
            })
    return rows


def prep_iemocap(out_dir: str):
    import json, urllib.request

    BASE_URL = "https://huggingface.co/datasets/Berzerker/IEMOCAP/resolve/main"
    os.makedirs(out_dir, exist_ok=True)

    for fname in ["train.txt", "test.txt"]:
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest):
            url = f"{BASE_URL}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)

    all_train = _parse_iemocap_txt(os.path.join(out_dir, "train.txt"))
    all_test  = _parse_iemocap_txt(os.path.join(out_dir, "test.txt"))

    dial_ids  = list(dict.fromkeys(r["Dialogue_ID"] for r in all_train))
    cut       = int(len(dial_ids) * 0.9)
    train_ids = set(dial_ids[:cut])
    val_ids   = set(dial_ids[cut:])

    split_rows = {
        "train": [r for r in all_train if r["Dialogue_ID"] in train_ids],
        "val":   [r for r in all_train if r["Dialogue_ID"] in val_ids],
        "test":  all_test,
    }

    for split_name, rows in split_rows.items():
        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)
        total_convs = len(set(r["Dialogue_ID"] for r in rows))
        print(f"  {split_name}: {total_convs} conversations, {len(rows)} utterances → {out_path}")

    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        json.dump(IEMOCAP_EMO_MAP, f, indent=2)
    print("  IEMOCAP prep done.")


# ── MultiDialog ───────────────────────────────────────────────────────────────
# Original labels: neutral, happy, fear, angry, disgusting, surprising, sad

def prep_multidialog(out_dir: str):
    from datasets import load_dataset
    import json

    print("Downloading IVLLab/MultiDialog ...")
    os.makedirs(out_dir, exist_ok=True)

    split_map = {
        "train": "train",
        "val":   "valid_freq",
        "test":  "test_freq",
    }

    for split_name, hf_split in split_map.items():
        # Each config's internal split name matches the config name
        ds = load_dataset("IVLLab/MultiDialog", hf_split, split=hf_split, trust_remote_code=True)
        rows = []
        for sample in ds:
            emo_name = str(sample.get("emotion", "neutral")).strip().lower()
            rows.append({
                "Dialogue_ID":  str(sample.get("conv_id", 0)),
                "Utterance_ID": str(sample.get("turn_id", 0)),
                "Utterance":    str(sample.get("value", "")).strip(),
                "Speaker":      str(sample.get("from", "spk_0")).strip(),
                "Emotion":      emo_name,
            })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)

        total_convs = len(set(r["Dialogue_ID"] for r in rows))
        print(f"  {split_name}: {total_convs} conversations, {len(rows)} utterances → {out_path}")

    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        unique_emos = sorted(set(r["Emotion"] for r in rows))
        json.dump({e: e for e in unique_emos}, f, indent=2)
    print("  MultiDialog prep done.")


# ── EmoryNLP ──────────────────────────────────────────────────────────────────
# Original labels: Neutral, Joyful, Peaceful, Powerful, Scared, Mad, Sad
# Stored lowercase as-is — no remapping.

EMORYNLP_BASE = (
    "https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json"
)


def prep_emorynlp(out_dir: str):
    import json, urllib.request

    os.makedirs(out_dir, exist_ok=True)

    file_map = {
        "train": "emotion-detection-trn.json",
        "val":   "emotion-detection-dev.json",
        "test":  "emotion-detection-tst.json",
    }

    for split_name, fname in file_map.items():
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest):
            url = f"{EMORYNLP_BASE}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)

        with open(dest, encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        episodes = data.get("episodes", []) if isinstance(data, dict) else data
        for episode in episodes:
            for scene in episode.get("scenes", []):
                scene_id   = scene.get("scene_id", "unknown")
                utterances = scene.get("utterances", [])
                for utt in utterances:
                    emo_name = str(utt.get("emotion", "Neutral")).strip().lower()
                    speakers = utt.get("speakers", ["unknown"])
                    speaker  = speakers[0] if speakers else "unknown"
                    rows.append({
                        "Dialogue_ID":  scene_id,
                        "Utterance_ID": utt.get("utterance_id", ""),
                        "Utterance":    utt.get("transcript", "").strip(),
                        "Speaker":      speaker,
                        "Emotion":      emo_name,
                    })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)

        total_convs = len(set(r["Dialogue_ID"] for r in rows))
        print(f"  {split_name}: {total_convs} scenes, {len(rows)} utterances → {out_path}")

    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        unique_emos = sorted(set(r["Emotion"] for r in rows))
        json.dump({e: e for e in unique_emos}, f, indent=2)
    print("  EmoryNLP prep done.")


# ── Appraisal Theory Dataset ──────────────────────────────────────────────────
# Combined CSV from https://github.com/shuiguolanzi397/Emotion-Prediction-Dataset
# Contains MELD + EmoryNLP + DailyDialog filtered for appraisal-rich dialogues
# (min 4 distinct emotions, min 2 emotion transitions per dialogue).
#
# Columns: Sr No., Is_Transition, Is_Appraisal_Driven, Utterance, Speaker,
#          Emotion, Dialogue_ID, Utterance_ID, Set (train/dev/test),
#          expectation, dataset, Moment_Utterance_ID
#
# Usage: prep_data.py --dataset appraisal --source meld   --out_dir ./data/appraisal_meld
#        prep_data.py --dataset appraisal --source emorynlp --out_dir ./data/appraisal_emorynlp
#        prep_data.py --dataset appraisal --source dailydialog --out_dir ./data/appraisal_dailydialog
#        prep_data.py --dataset appraisal --source all    --out_dir ./data/appraisal_all

APPRAISAL_CSV_URL = (
    "https://raw.githubusercontent.com/shuiguolanzi397/"
    "Emotion-Prediction-Dataset/main/data/DailyDialog_EmoryNLP_MELD_gpt4o_q1.csv"
)

# Map the dataset column values in the CSV to canonical source names
APPRAISAL_SOURCE_MAP = {
    "meld":        ["MELD", "meld"],
    "emorynlp":    ["EmoryNLP", "emorynlp", "Emorynlp"],
    "dailydialog": ["DailyDialog", "dailydialog", "Daily Dialog"],
}


def prep_appraisal(out_dir: str, source: str = "all"):
    """
    Download and split the appraisal theory combined CSV into train/val/test CSVs.

    source: one of 'meld', 'emorynlp', 'dailydialog', 'all'
    """
    import urllib.request
    import json
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    # Shared cache under the parent data/appraisal directory to avoid re-downloading
    shared_cache_dir = os.path.join(os.path.dirname(out_dir), "_cache")
    os.makedirs(shared_cache_dir, exist_ok=True)
    cache_path = os.path.join(shared_cache_dir, "appraisal_raw.csv")
    if not os.path.exists(cache_path):
        print(f"  Downloading appraisal dataset from GitHub ...")
        urllib.request.urlretrieve(APPRAISAL_CSV_URL, cache_path)
        print(f"  Saved to {cache_path}")
    else:
        print(f"  Using cached raw CSV: {cache_path}")

    df = pd.read_csv(cache_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Always show what's in the dataset column so mismatches are obvious
    if "dataset" in df.columns:
        from collections import Counter
        print(f"  dataset column values: {dict(Counter(df['dataset'].astype(str)).most_common())}")
    if "Set" in df.columns:
        from collections import Counter as C2
        print(f"  Set column values:     {dict(C2(df['Set'].astype(str)).most_common())}")

    # Filter by source — case-insensitive to handle variant spellings
    if source != "all":
        valid_lower = {v.lower() for v in APPRAISAL_SOURCE_MAP.get(source, [source])}
        mask = df["dataset"].astype(str).str.strip().str.lower().isin(valid_lower)
        df = df[mask].copy()
        print(f"  Filtered to source='{source}': {len(df)} rows")
        if len(df) == 0:
            print(f"  [ERROR] No rows matched source='{source}'. "
                  f"Check dataset column values above.")
            return

    # Prefix Dialogue_ID with source to avoid collisions in 'all' mode
    df["Dialogue_ID"] = df["dataset"].astype(str).str.lower().str.replace(" ", "_") \
                        + "_" + df["Dialogue_ID"].astype(str)

    # Map Set column: dev → val
    set_map = {"train": "train", "dev": "val", "test": "test",
               "Train": "train", "Dev": "val", "Test": "test"}
    df["_split"] = df["Set"].map(set_map).fillna("train")

    all_rows = []
    for split_name in ["train", "val", "test"]:
        subset = df[df["_split"] == split_name].copy()
        rows = []
        for _, row in subset.iterrows():
            emo = str(row["Emotion"]).strip().lower()
            if emo in ("unknown", "other", "xxx", "oth"):
                continue   # skip ambiguous labels
            rows.append({
                "Dialogue_ID":        str(row["Dialogue_ID"]),
                "Utterance_ID":       str(row["Utterance_ID"]),
                "Utterance":          str(row["Utterance"]).strip(),
                "Speaker":            str(row["Speaker"]).strip(),
                "Emotion":            emo,
                "Is_Transition":      int(row.get("Is_Transition", 0)),
                "Is_Appraisal_Driven":int(row.get("Is_Appraisal_Driven", 0)),
            })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion",
                                                    "Is_Transition", "Is_Appraisal_Driven"])
            writer.writeheader()
            writer.writerows(rows)

        total_convs = len(set(r["Dialogue_ID"] for r in rows))
        from collections import Counter
        emo_dist = Counter(r["Emotion"] for r in rows)
        print(f"  {split_name}: {total_convs} dialogues, {len(rows)} utterances, "
              f"{len(emo_dist)} emotions → {out_path}")
        print(f"    top emotions: {dict(emo_dist.most_common(5))}")
        all_rows.extend(rows)

    unique_emos = sorted(set(r["Emotion"] for r in all_rows))
    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        json.dump({e: e for e in unique_emos}, f, indent=2)
    print(f"  Appraisal ({source}) prep done. {len(unique_emos)} emotion classes: {unique_emos}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True,
                   choices=["dailydialog", "iemocap", "multidialog", "emorynlp", "appraisal"])
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--source",   default="all",
                   choices=["meld", "emorynlp", "dailydialog", "all"],
                   help="Which source dataset to extract (appraisal only)")
    args = p.parse_args()

    if args.dataset == "dailydialog":
        prep_dailydialog(args.out_dir)
    elif args.dataset == "iemocap":
        prep_iemocap(args.out_dir)
    elif args.dataset == "multidialog":
        prep_multidialog(args.out_dir)
    elif args.dataset == "emorynlp":
        prep_emorynlp(args.out_dir)
    elif args.dataset == "appraisal":
        prep_appraisal(args.out_dir, source=args.source)
