"""Datasets and collators for Qwen3 Stage 2 Agent-SFT/RLVR."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from trainer.agent_chat import build_assistant_only_example


class QwenAgentSFTDataset(Dataset):
    """JSONL Agent-SFT data with template-derived assistant-only labels."""

    def __init__(self, path: str | Path, tokenizer, max_length: int = 1536):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "conversations" not in row:
                    raise ValueError(
                        f"{self.path}:{line_number} is missing conversations"
                    )
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        encoded = build_assistant_only_example(
            self.tokenizer,
            self.rows[index]["conversations"],
            max_length=self.max_length,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class AssistantOnlyCollator:
    """Dynamically right-pad SFT batches while excluding padding from loss."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, rows):
        max_length = max(row["input_ids"].numel() for row in rows)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        def padded(row, key, value):
            tensor = row[key]
            return torch.cat(
                [tensor, tensor.new_full((max_length - tensor.numel(),), value)]
            )

        return {
            "input_ids": torch.stack(
                [padded(row, "input_ids", self.pad_token_id) for row in rows]
            ),
            "attention_mask": torch.stack(
                [padded(row, "attention_mask", 0) for row in rows]
            ),
            "labels": torch.stack([padded(row, "labels", -100) for row in rows]),
        }


def audit_sft_dataset(dataset: QwenAgentSFTDataset, samples: int | None = None) -> dict:
    """Return deterministic supervision statistics before expensive training."""

    count = min(len(dataset), samples) if samples else len(dataset)
    supervised_tokens = []
    sequence_lengths = []
    for index in range(count):
        row = dataset[index]
        supervised_tokens.append(int(row["labels"].ne(-100).sum().item()))
        sequence_lengths.append(int(row["input_ids"].numel()))
    return {
        "rows_checked": count,
        "zero_supervision_count": sum(value == 0 for value in supervised_tokens),
        "min_supervised_tokens": min(supervised_tokens, default=0),
        "max_supervised_tokens": max(supervised_tokens, default=0),
        "max_sequence_length": max(sequence_lengths, default=0),
    }
