"""Deterministic GSM8K normalization, splitting, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


SOURCE = "openai/gsm8k"
CONFIG = "main"


def extract_gold_answer(answer: str) -> str:
    """Return the canonical answer following GSM8K's final ``####`` marker."""

    if "####" not in answer:
        raise ValueError("GSM8K answer is missing the final '####' marker")
    value = answer.rsplit("####", 1)[1].strip().replace(",", "")
    if not value:
        raise ValueError("GSM8K final answer is empty")
    return value


def normalize_rows(rows: Iterable[Mapping[str, object]], split: str) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows):
        question, answer = str(row["question"]).strip(), str(row["answer"]).strip()
        if not question or not answer:
            raise ValueError(f"empty question/answer at {split}[{index}]")
        normalized.append({
            "id": f"gsm8k_{split}_{index:05d}",
            "question": question,
            "answer": answer,
            "gold_answer": extract_gold_answer(answer),
            "source": SOURCE,
            "config": CONFIG,
            "original_split": split,
        })
    return normalized


def split_train_validation(
    rows: Sequence[dict], validation_size: int | float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split official train deterministically without changing official test."""

    if isinstance(validation_size, float):
        if not 0 < validation_size < 1:
            raise ValueError("float validation_size must be between 0 and 1")
        count = round(len(rows) * validation_size)
    else:
        count = validation_size
    if not 0 < count < len(rows):
        raise ValueError("validation_size must leave at least one row in each split")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:count])
    train, validation = [], []
    for index, row in enumerate(rows):
        target = validation if index in validation_indices else train
        target.append({**row, "split": "validation" if index in validation_indices else "train"})
    return train, validation


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def assert_disjoint(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    seen_ids, seen_questions = {}, {}
    for split, rows in splits.items():
        for row in rows:
            row_id = str(row["id"])
            question_key = " ".join(str(row["question"]).casefold().split())
            if row_id in seen_ids:
                raise ValueError(f"duplicate id {row_id!r} in {seen_ids[row_id]} and {split}")
            if question_key in seen_questions:
                raise ValueError(
                    f"question leakage between {seen_questions[question_key]} and {split}: {row_id}"
                )
            seen_ids[row_id], seen_questions[question_key] = split, split


def build_manifest(paths: Mapping[str, str | Path], seed: int) -> dict:
    return {
        "dataset": SOURCE,
        "config": CONFIG,
        "seed": seed,
        "policy": "official train -> train/validation; official test -> test only",
        "splits": {
            split: {
                "path": str(Path(path)),
                "rows": sum(1 for _ in read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for split, path in paths.items()
        },
    }
