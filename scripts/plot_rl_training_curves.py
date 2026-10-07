"""Plot comparable RL training diagnostics from MiniMind JSONL metrics.

The horizontal axis is candidate prompt groups rather than optimizer steps.  This
matters for DAPO because dynamic sampling discards zero-variance groups and thus
performs fewer optimizer updates while consuming the same rollout budget.
"""

import argparse
import glob
import json
import math
import os

import matplotlib.pyplot as plt


PANELS = (
    ("policy_loss", "Policy loss", "linear"),
    ("reward", "Mean verifiable reward", "linear"),
    ("task_accuracy", "Task accuracy", "linear"),
    ("kl_k3", "KL (k3 estimator)", "log"),
    ("clip_fraction", "Clipped-token fraction", "linear"),
    ("group_reward_std", "Within-group reward std", "linear"),
)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_run(path):
    points = []
    config = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics", {})
            config = record.get("config", config)
            # The final budget record has no instantaneous training diagnostics.
            if finite(metrics.get("candidate_groups")) and any(
                finite(metrics.get(name)) for name, _, _ in PANELS
            ):
                points.append(metrics)
    if not points:
        raise ValueError(f"no plottable training records in {path}")
    return config.get("algorithm", os.path.basename(path)).upper(), points


def rolling_candidate_mean(xs, ys, span):
    """Trailing mean over a candidate-budget window, not a row-count window."""
    smoothed = []
    left = 0
    running_sum = 0.0
    running_count = 0
    for right, (x_value, y_value) in enumerate(zip(xs, ys)):
        if finite(y_value):
            running_sum += float(y_value)
            running_count += 1
        while left <= right and xs[left] < x_value - span:
            if finite(ys[left]):
                running_sum -= float(ys[left])
                running_count -= 1
            left += 1
        smoothed.append(running_sum / running_count if running_count else float("nan"))
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO-family training curves")
    parser.add_argument(
        "inputs", nargs="*", default=["out/metrics/gsm8k_*_full_s42.jsonl"],
        help="JSONL paths or glob patterns",
    )
    parser.add_argument("--output", default="out/plots/gsm8k_rl_training_curves.png")
    parser.add_argument("--window_groups", type=int, default=250)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.window_groups < 1:
        parser.error("--window_groups must be >= 1")

    paths = sorted({path for pattern in args.inputs for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No JSONL files matched")
    runs = [load_run(path) for path in paths]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    colors = {"GRPO": "#1f77b4", "CISPO": "#ff7f0e", "DAPO": "#2ca02c", "GSPO": "#d62728"}

    for axis, (metric, title, scale) in zip(axes.flat, PANELS):
        for algorithm, points in runs:
            xs = [float(point["candidate_groups"]) for point in points]
            ys = [float(point[metric]) if finite(point.get(metric)) else float("nan") for point in points]
            smooth = rolling_candidate_mean(xs, ys, args.window_groups)
            color = colors.get(algorithm)
            axis.plot(xs, ys, color=color, alpha=0.10, linewidth=0.7)
            axis.plot(xs, smooth, color=color, linewidth=2.0, label=algorithm)
        axis.set_title(title)
        axis.set_yscale(scale)
        axis.set_xlabel("Candidate prompt groups consumed")
        if metric in {"task_accuracy", "clip_fraction"}:
            axis.set_ylim(bottom=0)
        if metric == "policy_loss":
            axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    figure.suptitle(
        f"GSM8K Agentic-RL training diagnostics (trailing {args.window_groups}-candidate mean)",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    pdf_path = os.path.splitext(args.output)[0] + ".pdf"
    figure.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {args.output}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
