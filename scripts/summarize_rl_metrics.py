"""Aggregate MiniMind RL JSONL logs across algorithms and random seeds."""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict


REPORT_METRICS = (
    "reward", "task_accuracy", "answer_accuracy", "kl_k3",
    "local_reference_log_ratio_abs_mean", "local_kl_k3_p95", "local_kl_k3_max",
    "format_valid_rate", "tool_call_valid_rate",
    "tool_execution_success_rate", "required_tool_coverage_rate",
    "tool_evidence_coverage_rate",
    "avg_response_len", "p95_response_len", "clip_fraction",
    "rollout_logprob_mae", "rollout_ratio_mean",
    "group_reward_std", "zero_variance_group_rate", "unfinished_rate",
    "dynamic_acceptance_rate",
    "candidate_groups", "accepted_groups", "candidate_trajectories",
    "candidate_action_tokens", "tool_calls", "optimizer_updates",
    "wall_time_seconds",
)

CUMULATIVE_METRICS = {
    "candidate_groups", "accepted_groups", "candidate_trajectories",
    "candidate_action_tokens", "tool_calls", "optimizer_updates",
    "wall_time_seconds",
}


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Aggregate final-window RL training metrics")
    parser.add_argument("inputs", nargs="+", help="JSONL路径或glob")
    parser.add_argument("--last_n", type=int, default=20, help="每个run统计最后N条记录")
    parser.add_argument("--output", default="./out/metrics/algorithm_comparison.csv")
    args = parser.parse_args()

    paths = sorted({path for pattern in args.inputs for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No JSONL files matched")
    per_algorithm = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            continue
        records = records[-args.last_n:]
        config = records[-1].get("config", {})
        algorithm = config.get("algorithm", "unknown")
        seed = config.get("seed", "unknown")
        run = {"algorithm": algorithm, "seed": seed, "path": path}
        for metric in REPORT_METRICS:
            values = [record.get("metrics", {}).get(metric) for record in records]
            values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
            run[metric] = values[-1] if metric in CUMULATIVE_METRICS and values else mean(values)
            run[f"temporal_{metric}_std"] = (
                float("nan") if metric in CUMULATIVE_METRICS else sample_std(values)
            )
        per_algorithm[algorithm].append(run)

    rows = []
    for algorithm, runs in sorted(per_algorithm.items()):
        row = {"algorithm": algorithm, "runs": len(runs)}
        for metric in REPORT_METRICS:
            values = [run[metric] for run in runs if math.isfinite(run[metric])]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_seed_std"] = sample_std(values)
        reward_temporal = [run["temporal_reward_std"] for run in runs if math.isfinite(run["temporal_reward_std"])]
        row["reward_temporal_std_mean"] = mean(reward_temporal)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} from {len(paths)} runs")


if __name__ == "__main__":
    main()
