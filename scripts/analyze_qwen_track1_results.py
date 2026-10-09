"""Create paired statistics for the completed Stage 2 Track 1 official test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LABELS = ["base", "sft_only", "pure_grpo", "sft_grpo"]
PAIRWISE = [
    ("sft_only", "base"),
    ("pure_grpo", "base"),
    ("sft_grpo", "sft_only"),
    ("sft_grpo", "pure_grpo"),
]
METRICS = [
    "task_accuracy",
    "answer_accuracy",
    "format_valid_rate",
    "tool_call_valid_rate",
    "tool_execution_success_rate",
    "required_tool_coverage_rate",
    "tool_evidence_coverage_rate",
    "unfinished_rate",
    "avg_response_tokens",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", default="out/stage2/qwen3_4b/evaluation/test"
    )
    parser.add_argument(
        "--output-dir", default="out/stage2/qwen3_4b/analysis"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    logs = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(tail + 1)
    ]
    maximum = max(logs)
    lower_tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_interval(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if treatment.shape != control.shape:
        raise ValueError("paired bootstrap inputs have different shapes")
    differences = treatment.astype(np.float64) - control.astype(np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        count = min(1000, samples - offset)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        estimates[offset : offset + count] = differences[indices].mean(axis=1)
        offset += count
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main():
    args = parse_args()
    evaluation_dir = ROOT / args.evaluation_dir
    output_dir = ROOT / args.output_dir
    summary = read_json(evaluation_dir / "summary.json")
    manifest = read_json(evaluation_dir / "manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("official-test manifest is not COMPLETE")
    if [row["label"] for row in summary] != LABELS:
        raise RuntimeError("unexpected official-test label order")

    summary_by_label = {row["label"]: row for row in summary}
    trajectories: dict[str, dict[tuple[int, int], dict]] = {}
    for label in LABELS:
        rows = read_jsonl(evaluation_dir / f"{label}_trajectories.jsonl")
        keyed = {(int(row["seed"]), int(row["index"])): row for row in rows}
        if len(keyed) != len(rows):
            raise RuntimeError(f"duplicate trajectory key for {label}")
        trajectories[label] = keyed
    keys = sorted(trajectories[LABELS[0]])
    for label in LABELS[1:]:
        if sorted(trajectories[label]) != keys:
            raise RuntimeError(f"paired trajectory coverage differs for {label}")

    arm_results = []
    for label in LABELS:
        row = summary_by_label[label]
        successes = sum(bool(trajectories[label][key]["task_success"]) for key in keys)
        if not math.isclose(successes / len(keys), row["task_accuracy"]):
            raise RuntimeError(f"task accuracy mismatch for {label}")
        arm_results.append({
            "label": label,
            "num_trajectories": len(keys),
            "task_successes": successes,
            "task_accuracy": row["task_accuracy"],
            "task_accuracy_wilson_95": wilson_interval(successes, len(keys)),
            **{metric: row[metric] for metric in METRICS if metric != "task_accuracy"},
        })

    pairwise_results = []
    for pair_index, (treatment_label, control_label) in enumerate(PAIRWISE):
        treatment = np.asarray([
            bool(trajectories[treatment_label][key]["task_success"]) for key in keys
        ])
        control = np.asarray([
            bool(trajectories[control_label][key]["task_success"]) for key in keys
        ])
        treatment_only = int(np.logical_and(treatment, np.logical_not(control)).sum())
        control_only = int(np.logical_and(control, np.logical_not(treatment)).sum())
        pairwise_results.append({
            "treatment": treatment_label,
            "control": control_label,
            "task_accuracy_delta": float(treatment.mean() - control.mean()),
            "paired_bootstrap_delta_95": paired_bootstrap_interval(
                treatment,
                control,
                samples=args.bootstrap_samples,
                seed=args.seed + pair_index,
            ),
            "treatment_only_success": treatment_only,
            "control_only_success": control_only,
            "both_success": int(np.logical_and(treatment, control).sum()),
            "both_failure": int(
                np.logical_and(np.logical_not(treatment), np.logical_not(control)).sum()
            ),
            "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(
                treatment_only, control_only
            ),
        })

    report = {
        "status": "COMPLETE",
        "scope": (
            "Intervals treat the 1,319 official-test questions as the sampling "
            "unit. They do not quantify variation across training seeds; all "
            "Track 1 models use one training/decode seed (42)."
        ),
        "data_sha256": manifest["data_sha256"],
        "labels": LABELS,
        "num_paired_questions": len(keys),
        "bootstrap_samples": args.bootstrap_samples,
        "arms": arm_results,
        "pairwise": pairwise_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_test_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "official_test_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "label", "num_trajectories", "task_successes", *METRICS,
            "task_accuracy_ci_low", "task_accuracy_ci_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in arm_results:
            writer.writerow({
                **{field: row.get(field) for field in fieldnames},
                "task_accuracy_ci_low": row["task_accuracy_wilson_95"][0],
                "task_accuracy_ci_high": row["task_accuracy_wilson_95"][1],
            })

    markdown = [
        "# Stage 2 Track 1 官方测试结果",
        "",
        f"数据：GSM8K official test，共 {len(keys)} 题；所有臂使用 decode seed 42。",
        "",
        "| 模型 | 严格成功率 (95% Wilson CI) | 答案准确率 | 工具执行率 | 证据覆盖率 | 平均输出 token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in arm_results:
        low, high = row["task_accuracy_wilson_95"]
        markdown.append(
            f"| {row['label']} | {pct(row['task_accuracy'])} "
            f"[{pct(low)}, {pct(high)}] | {pct(row['answer_accuracy'])} | "
            f"{pct(row['tool_execution_success_rate'])} | "
            f"{pct(row['tool_evidence_coverage_rate'])} | "
            f"{row['avg_response_tokens']:.2f} |"
        )
    markdown.extend([
        "",
        "## 配对严格成功率比较",
        "",
        "| Treatment − Control | 差值 (95% paired bootstrap CI) | 仅 Treatment 成功 | 仅 Control 成功 | McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in pairwise_results:
        low, high = row["paired_bootstrap_delta_95"]
        markdown.append(
            f"| {row['treatment']} − {row['control']} | "
            f"{pct(row['task_accuracy_delta'])} [{pct(low)}, {pct(high)}] | "
            f"{row['treatment_only_success']} | {row['control_only_success']} | "
            f"{row['mcnemar_exact_two_sided_p']:.6g} |"
        )
    markdown.extend([
        "",
        "## 解释边界",
        "",
        "- 区间和 McNemar 检验以题目为抽样单位，只描述这一训练 seed 下的逐题差异。",
        "- 当前只有一个训练 seed 和一个 decode seed，不能把题目级显著性解释为跨训练重复的稳定性。",
        "- official test 仅用于最终报告，没有参与 reward、超参数或 checkpoint 选择。",
        "",
    ])
    (output_dir / "TRACK1_OFFICIAL_TEST_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
