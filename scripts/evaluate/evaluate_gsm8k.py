"""Score JSONL predictions with the shared GSM8K verifier."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.gsm8k import read_jsonl
from trainer.math_env import verify_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL with id and prediction fields")
    parser.add_argument("--references", required=True, help="prepared GSM8K split JSONL")
    parser.add_argument("--output")
    args = parser.parse_args()

    references = {row["id"]: row for row in read_jsonl(args.references)}
    predictions = list(read_jsonl(args.predictions))
    duplicate_ids = len(predictions) - len({row.get("id") for row in predictions})
    if duplicate_ids:
        raise ValueError(f"predictions contain {duplicate_ids} duplicate ids")
    unknown = [row.get("id") for row in predictions if row.get("id") not in references]
    if unknown:
        raise ValueError(f"unknown prediction ids: {unknown[:3]}")
    correct = sum(
        verify_answer(str(row.get("prediction", "")), references[row["id"]]["gold_answer"])
        for row in predictions
    )
    metrics = {
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "correct": correct,
        "predicted": len(predictions),
        "reference_rows": len(references),
        "coverage": len(predictions) / len(references) if references else 0.0,
    }
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
