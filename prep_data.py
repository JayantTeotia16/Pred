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
CANONICAL_EMOTIONS = ["neutral", "surprise", "fear", "joy", "sadness", "disgust", "anger"]
EMOTION2ID = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}

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


IEMOCAP_EMO_MAP = {
    "neu": "neutral",
    "ang": "anger",
    "sad": "sadness",
    "hap": "joy",
    "exc": "joy",       # excited → joy (standard IEMOCAP merge)
    "fru": "disgust",   # frustration → closest canonical class
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
      → speaker     = F or M  (second-to-last segment letter)
      → utt_id      = 000
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            utt_id_full, raw_emo, utterance = parts[0], parts[1].strip(), parts[2].strip()
            # Parse ID: e.g. Ses01F_impro01_F000
            segments = utt_id_full.rsplit("_", 1)   # ['Ses01F_impro01', 'F000']
            if len(segments) != 2:
                continue
            dial_id   = segments[0]                  # Ses01F_impro01
            spk_utt   = segments[1]                  # F000 or M000
            speaker   = spk_utt[0]                   # F or M
            utt_num   = spk_utt[1:]                  # 000

            emo_name  = IEMOCAP_EMO_MAP.get(raw_emo.lower(), "neutral")
            emo_int   = EMOTION2ID.get(emo_name, 0)

            rows.append({
                "Dialogue_ID":  dial_id,
                "Utterance_ID": utt_num,
                "Utterance":    utterance,
                "Speaker":      speaker,
                "Emotion":      emo_int,
            })
    return rows


def prep_iemocap(out_dir: str):
    import json, urllib.request

    BASE_URL = "https://huggingface.co/datasets/Berzerker/IEMOCAP/resolve/main"
    os.makedirs(out_dir, exist_ok=True)

    # Download train.txt and test.txt (valid.txt is empty)
    for fname in ["train.txt", "test.txt"]:
        dest = os.path.join(out_dir, fname)
        if not os.path.exists(dest):
            url = f"{BASE_URL}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)

    all_train = _parse_iemocap_txt(os.path.join(out_dir, "train.txt"))
    all_test  = _parse_iemocap_txt(os.path.join(out_dir, "test.txt"))

    # Split train → 90% train / 10% val by dialogue (keep conversations intact)
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
# Emotions: neutral, happy, fear, angry, disgusting→disgust, surprising→surprise, sad→sadness
MULTIDIALOG_EMO_MAP = {
    "neutral":    "neutral",
    "happy":      "joy",
    "fear":       "fear",
    "angry":      "anger",
    "disgusting": "disgust",
    "surprising": "surprise",
    "sad":        "sadness",
}


def prep_multidialog(out_dir: str):
    from datasets import load_dataset
    import json

    print("Downloading IVLLab/MultiDialog ...")
    os.makedirs(out_dir, exist_ok=True)

    # MultiDialog split names
    split_map = {
        "train": "train",
        "val":   "valid_freq",   # use frequent-emotion val set
        "test":  "test_freq",    # use frequent-emotion test set
    }

    for split_name, hf_split in split_map.items():
        ds = load_dataset("IVLLab/MultiDialog", split=hf_split, trust_remote_code=True)
        rows = []
        for sample in ds:
            raw_emo = str(sample.get("emotion", "neutral")).strip().lower()
            emo_name = MULTIDIALOG_EMO_MAP.get(raw_emo, "neutral")
            # Map canonical name → int via CANONICAL_EMOTIONS order
            emo_int = EMOTION2ID.get(emo_name, 0)
            rows.append({
                "Dialogue_ID":  str(sample.get("conv_id", 0)),
                "Utterance_ID": str(sample.get("turn_id", 0)),
                "Utterance":    str(sample.get("value", "")).strip(),
                "Speaker":      str(sample.get("from", "spk_0")).strip(),
                "Emotion":      emo_int,
            })

        out_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID",
                                                    "Utterance", "Speaker", "Emotion"])
            writer.writeheader()
            writer.writerows(rows)

        total_convs = len(set(r["Dialogue_ID"] for r in rows))
        print(f"  {split_name}: {total_convs} conversations, {len(rows)} utterances → {out_path}")

    # Emotion map for reference
    with open(os.path.join(out_dir, "emotion_map.json"), "w") as f:
        json.dump(MULTIDIALOG_EMO_MAP, f, indent=2)
    print("  MultiDialog prep done.")


# ── EmoryNLP ──────────────────────────────────────────────────────────────────
# 7 dispositional emotion labels — annotated at scene level per character,
# making this the strongest validation dataset for dispositional prediction.
EMORYNLP_EMO_MAP = {
    "neutral":  "neutral",
    "joyful":   "joy",
    "peaceful": "neutral",   # closest canonical
    "powerful": "joy",       # closest canonical (excited/energized)
    "scared":   "fear",
    "mad":      "anger",
    "sad":      "sadness",
}

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
        for dial_idx, episode in enumerate(data):
            # EmoryNLP structure: episode → scenes → utterances
            scenes = episode.get("scenes", [episode])  # fallback if flat
            for scene in scenes:
                scene_id  = scene.get("scene_id", str(dial_idx))
                utterances = scene.get("utterances", [])
                for utt in utterances:
                    raw_emo  = str(utt.get("emotion", "Neutral")).strip().lower()
                    emo_name = EMORYNLP_EMO_MAP.get(raw_emo, "neutral")
                    emo_int  = EMOTION2ID.get(emo_name, 0)
                    rows.append({
                        "Dialogue_ID":  scene_id,
                        "Utterance_ID": utt.get("utterance_id", ""),
                        "Utterance":    utt.get("transcript", "").strip(),
                        "Speaker":      utt.get("speaker",    "unknown").strip(),
                        "Emotion":      emo_int,
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
        json.dump(EMORYNLP_EMO_MAP, f, indent=2)
    print("  EmoryNLP prep done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True,
                   choices=["dailydialog", "iemocap", "multidialog", "emorynlp"])
    p.add_argument("--out_dir",  required=True)
    args = p.parse_args()

    if args.dataset == "dailydialog":
        prep_dailydialog(args.out_dir)
    elif args.dataset == "iemocap":
        prep_iemocap(args.out_dir)
    elif args.dataset == "multidialog":
        prep_multidialog(args.out_dir)
    elif args.dataset == "emorynlp":
        prep_emorynlp(args.out_dir)
