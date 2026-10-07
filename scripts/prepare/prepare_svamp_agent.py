"""Build a leakage-aware SVAMP-only train/holdout Agentic RL protocol.

The official SVAMP release is a 1,000-item challenge set, not an official
train/test dataset.  Its authors also publish five augmented cross-validation
folds.  We recover the 1,000 SVAMP rows from the five dev folds, use folds 0-3
as an 816-row derived train set, and freeze fold 4 as a 184-row holdout.  Fold
4 has no ``group_nums`` variation-family overlap with folds 0-3.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.gsm8k import sha256_file, write_jsonl  # noqa: E402
from scripts.prepare.prepare_gsm8k_agent_data import make_rl_row  # noqa: E402
from trainer.math_env import canonical_number, safe_calculate  # noqa: E402


NUMBER_TOKEN = re.compile(r"number(\d+)")
OPERATORS = {"+", "-", "*", "/"}


def display_path(path: Path) -> str:
    """Keep repository paths portable while allowing isolated /tmp audits."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def render_number_tokens(text: str, numbers: list[str]) -> str:
    def replace(match):
        index = int(match.group(1))
        if index >= len(numbers):
            raise ValueError(f"missing number{index} in {numbers}")
        return numbers[index]

    return NUMBER_TOKEN.sub(replace, text).strip()


def prefix_to_infix(prefix: str, numbers: list[str]) -> str:
    tokens = prefix.split()
    position = 0

    def parse() -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"truncated prefix equation: {prefix!r}")
        token = tokens[position]
        position += 1
        if token in OPERATORS:
            return f"({parse()} {token} {parse()})"
        match = NUMBER_TOKEN.fullmatch(token)
        if match:
            index = int(match.group(1))
            if index >= len(numbers):
                raise ValueError(f"missing {token} in {numbers}")
            return numbers[index]
        return token

    expression = parse()
    if position != len(tokens):
        raise ValueError(f"unused prefix tokens in {prefix!r}")
    return expression


def canonical_text_number(value) -> str:
    number = canonical_number(value)
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def convert_row(row: dict, fold: int, row_index: int) -> dict:
    numbers = row["Numbers"].split()
    body = render_number_tokens(row["Body"], numbers)
    question = render_number_tokens(row["Ques"], numbers)
    expression = prefix_to_infix(row["Equation"], numbers)
    result = safe_calculate(expression)["result"]
    gold = canonical_text_number(row["Answer"])
    if canonical_number(result) != canonical_number(gold):
        raise ValueError(
            f"SVAMP oracle mismatch fold={fold} row={row_index}: "
            f"{expression} -> {result}, expected {gold}"
        )
    source = {
        "id": f"svamp_fold{fold}_{row_index:04d}",
        "question": f"{body} {question}".strip(),
        "answer": f"<<{expression}={result}>>\n#### {gold}",
        "gold_answer": gold,
        "split": "derived_holdout" if fold == 4 else "derived_train",
    }
    converted = make_rl_row(source)
    return {
        **converted,
        "svamp_fold": fold,
        "svamp_group_nums": row["group_nums"],
        "svamp_type": row.get("Type") or "unknown",
        "svamp_prefix_equation": row["Equation"],
        "svamp_infix_expression": expression,
    }


def build_svamp_rows(raw_dir: Path, probe_size: int = 128) -> dict[str, list[dict]]:
    by_fold: dict[int, list[dict]] = {}
    all_questions = set()
    for fold in range(5):
        path = raw_dir / f"fold{fold}_dev.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        converted = []
        for row_index, row in enumerate(raw_rows):
            item = convert_row(row, fold, row_index)
            question = item["conversations"][1]["content"]
            if question in all_questions:
                raise RuntimeError(f"duplicate SVAMP question: {question}")
            all_questions.add(question)
            converted.append(item)
        by_fold[fold] = converted

    train = [row for fold in range(4) for row in by_fold[fold]]
    holdout = by_fold[4]
    train_groups = {row["svamp_group_nums"] for row in train}
    holdout_groups = {row["svamp_group_nums"] for row in holdout}
    overlap = train_groups & holdout_groups
    if overlap:
        raise RuntimeError(f"SVAMP variation-family leakage: {sorted(overlap)[:5]}")
    if len(train) != 816 or len(holdout) != 184:
        raise RuntimeError(
            f"unexpected SVAMP fold sizes: train={len(train)}, holdout={len(holdout)}"
        )
    if not 0 < probe_size <= len(train):
        raise ValueError("probe size must be in [1, len(train)]")
    # The first N rows are deterministic across the pinned source revision.
    probe = train[:probe_size]
    return {"train_rl": train, "holdout_rl": holdout, "probe_rl": probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/svamp")
    parser.add_argument("--output-dir", default="data/processed/svamp_agent")
    parser.add_argument("--manifest", default="dataset/manifests/svamp_agent.json")
    parser.add_argument("--probe-size", type=int, default=128)
    args = parser.parse_args()
    raw_dir = ROOT / args.raw_dir
    output_dir = ROOT / args.output_dir
    rows = build_svamp_rows(raw_dir, args.probe_size)
    paths = {}
    for name, values in rows.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        paths[name] = path

    source_paths = {
        f"fold{fold}_dev": raw_dir / f"fold{fold}_dev.csv" for fold in range(5)
    }
    manifest = {
        "dataset": "arkilpatel/SVAMP",
        "source_revision": "78e727689e1c1bebfc4be39c446898e8e10b0518",
        "protocol": "svamp_only_author_cv_folds_0_3_train_fold4_holdout",
        "official_split_warning": (
            "SVAMP is officially a 1,000-item challenge set. This experiment "
            "derives a train/holdout protocol from the authors' augmented CV folds."
        ),
        "policy": (
            "SVAMP rows in author folds 0-3 form train; fold 4 is frozen holdout; "
            "group_nums variation families are disjoint; holdout is verifier-only"
        ),
        "source": {
            name: {"path": display_path(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "files": {
            name: {
                "path": display_path(path),
                "rows": len(rows[name]),
                "sha256": sha256_file(path),
                "type_counts": dict(Counter(row["svamp_type"] for row in rows[name])),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
