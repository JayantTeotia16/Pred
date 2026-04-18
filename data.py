"""
data.py — Universal dataset loader.

Supports any dataset that has:
    - A sequence of utterances per conversation (dialogue_id groups them)
    - Per-utterance emotion labels (integer)
    - Optionally: speaker labels, utterance ordering, season info

Out of the box: eusip/silicone meld_e (text-only MELD, auto-downloaded).
Any HuggingFace dataset or local CSV can be used by changing DataConfig.

Speaker handling:
    - If speaker_col is set: speakers are tracked by their string label
      within each conversation (e.g. "Ross", "Rachel", or "spk_0").
    - If speaker_col is None: speaker index is inferred from turn parity
      (turn 0 → speaker 0, turn 1 → speaker 1, etc.) — works for dyadic.
    - Speaker IDs are LOCAL to each conversation, not global.
      The model never needs a global speaker vocabulary.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset

from config import ModelConfig, DataConfig, TrainingConfig


# ── Canonical emotion mapping (internal to this codebase) ────────────────────
# Datasets map their own integers → canonical names via DataConfig.emotion_int_to_name
# Then we re-index to these canonical IDs.
CANONICAL_EMOTIONS = [
    "neutral", "surprise", "fear", "joy", "sadness", "disgust", "anger"
]
EMOTION2ID = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
ID2EMOTION = {i: e for i, e in enumerate(CANONICAL_EMOTIONS)}


# ── Single conversation ───────────────────────────────────────────────────────

class Conversation:
    """
    One conversation as an ordered list of turns.
    Speaker IDs are LOCAL (0, 1, 2, ...) — assigned by order of first appearance.
    This makes the data loader fully dataset-agnostic.
    """

    def __init__(self, dialogue_id: str, season: int = 1):
        self.dialogue_id   = dialogue_id
        self.season        = season
        self.turns: List[Dict] = []
        self._speaker_map: Dict[str, int] = {}   # local speaker str → local int

    def _get_speaker_id(self, speaker_str: str) -> int:
        if speaker_str not in self._speaker_map:
            self._speaker_map[speaker_str] = len(self._speaker_map)
        return self._speaker_map[speaker_str]

    def add_turn(self, utterance: str, emotion_id: int, speaker_str: str = "spk"):
        self.turns.append({
            "utterance":  utterance,
            "speaker_id": self._get_speaker_id(speaker_str),
            "emotion_id": emotion_id,
        })

    def num_local_speakers(self) -> int:
        return len(self._speaker_map)

    def __len__(self):
        return len(self.turns)


# ── Universal dataset ─────────────────────────────────────────────────────────

class ConversationDataset(Dataset):
    """
    Dataset-agnostic conversation loader.

    Returns one conversation per item:
        input_ids       (T, L)  — tokenised utterances
        attention_mask  (T, L)
        speaker_ids     (T,)    — local speaker IDs (0,1,2,...)
        emotion_ids     (T,)    — canonical emotion IDs; -1 = padding
        length          int     — actual turns (rest is padding)
        dialogue_id     str
        season          int
        num_speakers    int     — number of distinct speakers in this conversation

    Padding: conversations shorter than max_conversation_length are
    padded with zeros / -1 so batches can be stacked.
    """

    def __init__(
        self,
        split: str,                    # "train" | "val" | "test"
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        train_cfg: TrainingConfig,
    ):
        self.model_cfg  = model_cfg
        self.data_cfg   = data_cfg
        self.max_turns  = train_cfg.max_conversation_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.llama_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load raw data
        raw = self._load_split(split)

        # Build conversations and emotion vocab
        self.conversations, self.emotion2id = self._build_conversations(raw, data_cfg)

        # Update model config with actual emotion count
        model_cfg.num_emotions  = len(self.emotion2id)
        model_cfg.emotion_labels = [
            k for k, _ in sorted(self.emotion2id.items(), key=lambda x: x[1])
        ]

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_split(self, split: str) -> List[Dict]:
        """Load raw rows from HuggingFace or local CSV, return list of dicts."""
        dc = self.data_cfg

        # Map split name to dataset split
        split_map = {"train": dc.train_split, "val": dc.val_split, "test": dc.test_split}
        hf_split  = split_map[split]

        if dc.hf_dataset_name is not None:
            ds = hf_load_dataset(dc.hf_dataset_name, dc.hf_config_name, split=hf_split, trust_remote_code=True)
            rows = [dict(row) for row in ds]
        else:
            import pandas as pd
            file_map = {"train": dc.train_file, "val": dc.val_file, "test": dc.test_file}
            path     = os.path.join(dc.local_data_dir, file_map[split])
            df       = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            rows     = df.to_dict("records")

        return rows

    # ── Build conversations ───────────────────────────────────────────────

    def _build_conversations(
        self,
        rows: List[Dict],
        dc: DataConfig,
    ) -> Tuple[List[Conversation], Dict[str, int]]:
        """
        Groups rows by dialogue_id, sorts turns, builds Conversation objects.
        Derives a canonical emotion2id mapping from DataConfig.
        """
        # ── Emotion mapping ───────────────────────────────────────────────
        if dc.emotion_int_to_name is not None:
            # Use the provided mapping
            int_to_name = {i: name for i, name in enumerate(dc.emotion_int_to_name)}
        else:
            # Auto-detect: collect all unique emotion values from data
            unique = sorted(set(str(r[dc.emotion_col]) for r in rows))
            int_to_name = {i: v for i, v in enumerate(unique)}

        # Build canonical emotion2id: name → internal ID
        # We align to CANONICAL_EMOTIONS where possible, append unknowns
        emotion2id: Dict[str, int] = {}
        for name in CANONICAL_EMOTIONS:
            if name in int_to_name.values():
                emotion2id[name] = len(emotion2id)
        for name in int_to_name.values():
            if name not in emotion2id:
                emotion2id[name] = len(emotion2id)

        # ── Group by dialogue ─────────────────────────────────────────────
        dialogue_rows: Dict[str, List] = defaultdict(list)
        for r in rows:
            did = str(r[dc.dialogue_id_col])
            dialogue_rows[did].append(r)

        # Sort turns within each dialogue
        def sort_key(r):
            if dc.utterance_id_col and dc.utterance_id_col in r:
                uid = r[dc.utterance_id_col]
                try:
                    return int(uid)
                except (ValueError, TypeError):
                    return str(uid)
            return 0

        for did in dialogue_rows:
            dialogue_rows[did].sort(key=sort_key)

        # ── Infer season proxy from dialogue_id order ─────────────────────
        # Approximate: split dialogue IDs into 10 equal buckets
        all_ids = sorted(dialogue_rows.keys(),
                         key=lambda x: int(x) if x.isdigit() else x)
        n = max(len(all_ids), 1)
        def season_proxy(did: str) -> int:
            idx = all_ids.index(did) if did in all_ids else 0
            return max(1, min(10, int(idx / n * 10) + 1))

        # ── Build Conversation objects ────────────────────────────────────
        conversations = []
        for did, d_rows in dialogue_rows.items():
            conv = Conversation(did, season=season_proxy(did))
            for i, r in enumerate(d_rows):
                utterance = str(r[dc.utterance_col])

                # Emotion
                raw_emo = r[dc.emotion_col]
                if isinstance(raw_emo, int):
                    emo_name = int_to_name.get(raw_emo, "neutral")
                else:
                    emo_name = str(raw_emo).strip().lower()
                emo_id = emotion2id.get(emo_name, 0)

                # Speaker — local within conversation
                if dc.speaker_col and dc.speaker_col in r and r[dc.speaker_col]:
                    speaker_str = str(r[dc.speaker_col]).strip()
                else:
                    # No speaker info: alternate between "spk_0"/"spk_1" for dyadic
                    speaker_str = f"spk_{i % 2}"

                conv.add_turn(utterance, emo_id, speaker_str)
                if len(conv) >= self.max_turns:
                    break

            if len(conv) >= 2:   # need at least 2 turns for prediction task
                conversations.append(conv)

        print(f"  Loaded {len(conversations)} conversations, "
              f"{sum(len(c) for c in conversations)} utterances, "
              f"{len(emotion2id)} emotion classes.")
        return conversations, emotion2id

    # ── Tokenise ──────────────────────────────────────────────────────────

    def _tokenise(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.model_cfg.llama_max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        conv  = self.conversations[idx]
        T     = len(conv)
        texts = [t["utterance"]  for t in conv.turns]

        input_ids, attention_mask = self._tokenise(texts)   # (T, L)

        speaker_ids = torch.tensor(
            [t["speaker_id"] for t in conv.turns], dtype=torch.long
        )
        emotion_ids = torch.tensor(
            [t["emotion_id"] for t in conv.turns], dtype=torch.long
        )

        # Pad to max_turns
        pad = self.max_turns - T
        if pad > 0:
            L = input_ids.shape[1]
            input_ids      = torch.cat([input_ids,      torch.zeros(pad, L, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, L, dtype=torch.long)], dim=0)
            speaker_ids    = torch.cat([speaker_ids,    torch.zeros(pad,    dtype=torch.long)], dim=0)
            emotion_ids    = torch.cat([emotion_ids,    torch.full((pad,), -1, dtype=torch.long)], dim=0)

        return {
            "input_ids":      input_ids,          # (max_turns, L)
            "attention_mask": attention_mask,      # (max_turns, L)
            "speaker_ids":    speaker_ids,         # (max_turns,) — LOCAL IDs
            "emotion_ids":    emotion_ids,         # (max_turns,)
            "length":         torch.tensor(T),
            "num_speakers":   torch.tensor(conv.num_local_speakers()),
            "dialogue_id":    conv.dialogue_id,
            "season":         torch.tensor(conv.season),
        }


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_dataloaders(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
) -> Tuple["ConversationDataset", "ConversationDataset", "ConversationDataset",
           DataLoader, DataLoader, DataLoader]:

    print("Loading train split...")
    train_ds = ConversationDataset("train", model_cfg, data_cfg, train_cfg)

    print("Loading val split...")
    val_ds   = ConversationDataset("val",   model_cfg, data_cfg, train_cfg)

    print("Loading test split...")
    test_ds  = ConversationDataset("test",  model_cfg, data_cfg, train_cfg)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,          # safer for HF datasets
            pin_memory=True,
        )

    return (
        train_ds,
        val_ds,
        test_ds,
        make_loader(train_ds, True),
        make_loader(val_ds,   False),
        make_loader(test_ds,  False),
    )
