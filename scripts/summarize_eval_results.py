"""Aggregate fixed-set Agent RLVR results without treating decode seeds as runs."""

import argparse
import csv
import math
import os
import re
import statistics
from collections import defaultdict


METRICS = (
    "reward", "task_accuracy", "answer_accuracy", "format_valid_rate",
    "tool_call_valid_rate", "tool_execution_success_rate", "kl_k3",
    "required_tool_coverage_rate",
    "tool_evidence_coverage_rate",
    "response_length", "p95_response_length", "unfinished", "tokens_per_second",
)


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def algorithm_name(checkpoint):
    match = re.search(r"(?:^|_)(grpo|cispo|dapo|gspo)(?:_|$)", checkpoint.lower())
    return match.group(1) if match else checkpoint


def main():
    parser = argparse.ArgumentParser(description="Aggregate RLVR eval CSV across training seeds")
    parser.add_argument("input", help="summary.csv from eval_agent_rlvr.py")
    parser.add_argument("--output", default="./out/eval_rlvr/algorithm_comparison.csv")
    parser.add_argument(
        "--baseline", default="agent_sft",
        help="Checkpoint/algorithm used for delta columns (default: agent_sft)",
    )
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Evaluation summary is empty")

    # Decode seeds are repeated measurements of one trained checkpoint.  First
    # average them within checkpoint, then use checkpoints (training seeds) as
    # the independent runs for the reported standard deviation.
    per_checkpoint = defaultdict(list)
    for row in rows:
        per_checkpoint[row["checkpoint"]].append(row)
    per_algorithm = defaultdict(list)
    for checkpoint, checkpoint_rows in per_checkpoint.items():
        run = {"checkpoint": checkpoint}
        for metric in METRICS:
            values = [float(row[metric]) for row in checkpoint_rows if row.get(metric)]
            run[metric] = mean([value for value in values if math.isfinite(value)])
        per_algorithm[algorithm_name(checkpoint)].append(run)

    output_rows = []
    for algorithm, runs in sorted(per_algorithm.items()):
        row = {"algorithm": algorithm, "training_runs": len(runs)}
        for metric in METRICS:
            values = [run[metric] for run in runs if math.isfinite(run[metric])]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_training_seed_std"] = sample_std(values)
        output_rows.append(row)

    baseline_algorithm = algorithm_name(args.baseline)
    baseline = next(
        (row for row in output_rows if row["algorithm"] == baseline_algorithm),
        None,
    )
    if baseline is None:
        raise SystemExit(
            f"Missing baseline {baseline_algorithm!r} in evaluation summary"
        )
    delta_suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", baseline_algorithm)
    for row in output_rows:
        for metric in METRICS:
            value = row[f"{metric}_mean"]
            baseline_value = baseline[f"{metric}_mean"]
            row[f"{metric}_delta_vs_{delta_suffix}"] = (
                value - baseline_value
                if math.isfinite(value) and math.isfinite(baseline_value)
                else float("nan")
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {args.output} from {len(per_checkpoint)} trained checkpoints")


if __name__ == "__main__":
    main()
