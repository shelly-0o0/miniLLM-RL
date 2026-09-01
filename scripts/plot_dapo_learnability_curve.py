"""Join DAPO budget logs with fixed-set evaluations into a token curve."""

import argparse
import csv
import json
import os
from collections import defaultdict


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_eval(path):
    values = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            checkpoint = row["checkpoint"]
            try:
                group = int(checkpoint.rsplit("_groups", 1)[1])
            except (IndexError, ValueError):
                continue
            values[group].append(float(row["task_accuracy"]))
    return {group: sum(scores) / len(scores) for group, scores in values.items()}


def main():
    parser = argparse.ArgumentParser(description="Build DAPO learnability token curve")
    parser.add_argument("--metrics", required=True, help="training JSONL metrics")
    parser.add_argument("--train_eval", required=True, help="diagnostic-train summary.csv")
    parser.add_argument("--holdout_eval", required=True, help="holdout summary.csv")
    parser.add_argument("--output", required=True, help="output curve.csv")
    parser.add_argument("--plot", default=None, help="optional PNG path")
    args = parser.parse_args()

    budget_by_group = {}
    for record in read_jsonl(args.metrics):
        metrics = record.get("metrics", {})
        group = metrics.get("effective_groups")
        tokens = metrics.get("generated_tokens", metrics.get("candidate_action_tokens"))
        if group is not None and tokens is not None:
            budget_by_group[int(group)] = {
                "generated_tokens": int(tokens),
                "optimizer_updates": int(metrics.get("optimizer_updates", 0)),
                "candidate_trajectories": int(metrics.get("candidate_trajectories", 0)),
            }
    train_accuracy = read_eval(args.train_eval)
    holdout_accuracy = read_eval(args.holdout_eval)
    groups = sorted(set(train_accuracy) | set(holdout_accuracy) | {0})
    rows = []
    previous_budget = {"generated_tokens": 0, "optimizer_updates": 0, "candidate_trajectories": 0}
    for group in groups:
        if group in budget_by_group:
            previous_budget = budget_by_group[group]
        rows.append({
            "effective_groups": group,
            **previous_budget,
            "train_task_accuracy": train_accuracy.get(group, ""),
            "holdout_task_accuracy": holdout_accuracy.get(group, ""),
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            x = [float(row["generated_tokens"]) for row in rows]
            train_y = [float(row["train_task_accuracy"] or 0) for row in rows]
            holdout_y = [float(row["holdout_task_accuracy"] or 0) for row in rows]
            plt.plot(x, train_y, marker="o", label="diagnostic train")
            plt.plot(x, holdout_y, marker="o", label="holdout dev")
            plt.xlabel("Cumulative assistant generated tokens")
            plt.ylabel("Strict task success rate")
            plt.ylim(0, 1)
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot, dpi=160)
            plt.close()
        except (ImportError, TypeError, ValueError) as error:
            print(f"matplotlib unavailable or incompatible ({error}); wrote CSV only")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
