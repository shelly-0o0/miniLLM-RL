"""Build the disjoint A/B data protocol for the Qwen3 Track 2 study.

The official 7,473-row GSM8K training split is partitioned exactly once into
balanced, question-disjoint A and B pools.  Both pools retain every prompt for
online RLVR.  Agent-SFT files contain only rows whose calculator annotations
can be replayed and verified.  The official 1,319-row test split remains
untouched and is never used for optimization.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.gsm8k import assert_disjoint, read_jsonl, sha256_file, write_jsonl
from scripts.prepare.prepare_gsm8k_agent_data import make_rl_row, make_sft_row


def split_ab(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Return a deterministic, balanced A/B split while preserving row order."""

    if len(rows) < 2:
        raise ValueError("Track 2 requires at least two training rows")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    a_indices = set(indices[: len(rows) // 2])
    a_rows, b_rows = [], []
    for index, row in enumerate(rows):
        split = "track2_a" if index in a_indices else "track2_b"
        target = a_rows if index in a_indices else b_rows
        target.append({**row, "split": split, "track2_split": split[-1]})
    return a_rows, b_rows


def select_probe(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Select a stable diagnostic subset without changing training membership."""

    if not 0 < size <= len(rows):
        raise ValueError("probe size must be positive and no larger than the pool")
    selected = set(random.Random(seed).sample(range(len(rows)), size))
    return [row for index, row in enumerate(rows) if index in selected]


def select_matched(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Select a deterministic equal-budget comparison subset."""

    if size == len(rows):
        return list(rows)
    return select_probe(rows, size, seed)


def add_audit_fields(converted: dict, source: dict) -> dict:
    return {
        **converted,
        "source_split": source["split"],
        "track2_split": source["track2_split"],
    }


def build_track2_rows(
    official_train: list[dict],
    official_test: list[dict],
    *,
    split_seed: int,
    probe_seed: int,
    probe_size: int,
) -> dict[str, list[dict]]:
    a_source, b_source = split_ab(official_train, seed=split_seed)
    test_source = [
        {**row, "split": "test", "track2_split": "test"}
        for row in official_test
    ]
    assert_disjoint({"a": a_source, "b": b_source, "test": test_source})

    a_rl = [add_audit_fields(make_rl_row(row), row) for row in a_source]
    b_rl = [add_audit_fields(make_rl_row(row), row) for row in b_source]
    test_rl = [add_audit_fields(make_rl_row(row), row) for row in test_source]

    def sft_rows(source_rows):
        output = []
        for row in source_rows:
            converted = make_sft_row(row)
            if converted is not None:
                output.append(add_audit_fields(converted, row))
        return output

    a_sft = sft_rows(a_source)
    b_sft = sft_rows(b_source)
    a_sft_ids = {row["id"] for row in a_sft}
    b_sft_ids = {row["id"] for row in b_sft}
    a_verified_rl = [row for row in a_rl if row["id"] in a_sft_ids]
    b_verified_rl = [row for row in b_rl if row["id"] in b_sft_ids]
    comparison_size = min(len(a_verified_rl), len(b_verified_rl))
    a_compare_rl = select_matched(a_verified_rl, comparison_size, split_seed + 1000)
    b_compare_rl = select_matched(b_verified_rl, comparison_size, split_seed + 1001)

    return {
        "a_source": a_source,
        "b_source": b_source,
        "test_source": test_source,
        "a_rl": a_rl,
        "b_rl": b_rl,
        "a_sft": a_sft,
        "b_sft": b_sft,
        "a_compare_rl": a_compare_rl,
        "b_compare_rl": b_compare_rl,
        "a_probe_rl": select_probe(a_compare_rl, probe_size, probe_seed),
        "b_probe_rl": select_probe(b_compare_rl, probe_size, probe_seed + 1),
        "test_rl": test_rl,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare disjoint GSM8K Track 2 A/B data")
    parser.add_argument("--raw-dir", default="data/raw/gsm8k")
    parser.add_argument("--output-dir", default="data/processed/gsm8k_track2")
    parser.add_argument("--manifest", default="dataset/manifests/gsm8k_track2.json")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=31415)
    parser.add_argument("--probe-size", type=int, default=128)
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    output_dir = ROOT / args.output_dir
    official_train = list(read_jsonl(raw_dir / "official_train.jsonl"))
    official_test = list(read_jsonl(raw_dir / "official_test.jsonl"))
    rows = build_track2_rows(
        official_train,
        official_test,
        split_seed=args.split_seed,
        probe_seed=args.probe_seed,
        probe_size=args.probe_size,
    )

    paths = {}
    for name, values in rows.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        paths[name] = path

    manifest = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "protocol": "track2_disjoint_ab_agentic_rl",
        "split_seed": args.split_seed,
        "probe_seed": args.probe_seed,
        "probe_size": args.probe_size,
        "policy": (
            "official train is partitioned into mutually exclusive A/B pools; "
            "official test is evaluation-only; probe rows remain members of their "
            "training pools and are diagnostics, not a holdout"
        ),
        "source": {
            "official_train": {
                "path": str((raw_dir / "official_train.jsonl").relative_to(ROOT)),
                "rows": len(official_train),
                "sha256": sha256_file(raw_dir / "official_train.jsonl"),
            },
            "official_test": {
                "path": str((raw_dir / "official_test.jsonl").relative_to(ROOT)),
                "rows": len(official_test),
                "sha256": sha256_file(raw_dir / "official_test.jsonl"),
            },
        },
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "rows": len(rows[name]),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
