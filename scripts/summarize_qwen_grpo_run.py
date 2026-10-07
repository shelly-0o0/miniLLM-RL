"""Summarize a running Stage 2 GRPO JSONL log without touching the job."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIELDS = (
    "reward",
    "task_accuracy",
    "answer_accuracy",
    "format_valid_rate",
    "tool_call_valid_rate",
    "tool_execution_success_rate",
    "required_tool_coverage_rate",
    "tool_evidence_coverage_rate",
    "protocol_progress",
    "group_reward_std",
    "zero_variance_group",
    "kl_k3",
    "clip_fraction",
    "rollout_logprob_mae",
    "action_tokens",
    "unfinished_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--max-kl", type=float, default=10.0)
    parser.add_argument("--max-logprob-mae", type=float, default=0.1)
    return parser.parse_args()


def mean(rows, field):
    values = [float(row[field]) for row in rows if field in row]
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    path = Path(args.metrics)
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    rejections = []
    final = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "train" and isinstance(row.get("metrics"), dict):
                records.append(row["metrics"])
            if (
                row.get("split") == "safety_rejection"
                and isinstance(row.get("metrics"), dict)
            ):
                rejections.append(row["metrics"])
            if row.get("split") == "train" and isinstance(row.get("final"), dict):
                final = row["final"]
    if not records:
        print(json.dumps({"status": "waiting_for_first_group", "metrics": str(path)}))
        return
    window = records[-args.window :]
    numeric_values = [
        float(value)
        for row in records
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    finite = all(math.isfinite(value) for value in numeric_values)
    max_kl = max(float(row.get("kl_k3", 0.0)) for row in records)
    max_mae = max(float(row.get("rollout_logprob_mae", 0.0)) for row in records)
    last = records[-1]
    summary = {
        "status": "complete" if final else "running",
        "metrics": str(path),
        "records": len(records),
        "safety_rejections": len(rejections),
        "last_safety_rejection": rejections[-1] if rejections else None,
        "last_candidate_group": int(last["candidate_group"]),
        "last_optimizer_updates": int(last["optimizer_updates"]),
        "last_wall_time_seconds": float(last["wall_time_seconds"]),
        "window": len(window),
        "window_means": {field: mean(window, field) for field in FIELDS},
        "guards": {
            "all_numeric_finite": finite,
            "max_kl_k3": max_kl,
            "max_kl_threshold": args.max_kl,
            "kl_ok": finite and max_kl <= args.max_kl,
            "max_rollout_logprob_mae": max_mae,
            "max_logprob_mae_threshold": args.max_logprob_mae,
            "rollout_logprob_mae_ok": finite and max_mae <= args.max_logprob_mae,
        },
        "final": final,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
