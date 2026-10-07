"""Produce the final Stage-1 GSM8K validation/test evidence bundle.

The unit of replication is a trained checkpoint, not a decode seed.  Decode
seeds are averaged within a checkpoint; means and sample standard deviations
are then computed across independent training seeds.  Paired bootstrap
intervals resample GSM8K questions and therefore describe item-sampling
uncertainty only.  Training-seed variability is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ALGORITHMS = ("grpo", "cispo", "dapo", "gspo")
TRAIN_SEEDS = (42, 43, 44)
DECODE_SEEDS = (42, 43, 44)
METRICS = (
    "reward",
    "task_accuracy",
    "answer_accuracy",
    "format_valid_rate",
    "tool_call_valid_rate",
    "tool_execution_success_rate",
    "required_tool_coverage_rate",
    "tool_evidence_coverage_rate",
    "unfinished",
    "response_length",
    "p95_response_length",
    "kl_k3",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def algorithm_name(checkpoint: str) -> str:
    if checkpoint in {"agent_sft", "full_sft"}:
        return checkpoint
    match = re.search(r"(?:^|_)(grpo|cispo|dapo|gspo)(?:_|$)", checkpoint.lower())
    if not match:
        raise RuntimeError(f"Cannot identify algorithm from checkpoint: {checkpoint}")
    return match.group(1)


def checkpoint_train_seed(checkpoint: str) -> int | None:
    match = re.search(r"_s(\d+)$", checkpoint)
    return int(match.group(1)) if match else None


def validation_sources(root: Path) -> list[tuple[Path, Path]]:
    sources = []
    for algorithm in ALGORITHMS:
        for train_seed in TRAIN_SEEDS:
            directory = root / algorithm if train_seed == 42 else root / algorithm / f"train_seed_{train_seed}"
            sources.append((directory / "summary.csv", directory / "trajectories.jsonl"))
    return sources


def collect_validation(repo: Path) -> tuple[list[dict], list[dict]]:
    baseline_dir = repo / "out/eval_gsm8k_agent_sft_validation"
    summaries = [
        row for row in read_csv(baseline_dir / "summary.csv")
        if row["checkpoint"] in {"full_sft", "agent_sft"}
    ]
    trajectories = [
        row for row in read_jsonl(baseline_dir / "trajectories.jsonl")
        if row["checkpoint"] in {"full_sft", "agent_sft"}
    ]
    for summary_path, trajectory_path in validation_sources(repo / "out/eval_full"):
        summaries.extend(read_csv(summary_path))
        trajectories.extend(read_jsonl(trajectory_path))
    return summaries, trajectories


def collect_test(repo: Path, winner: str) -> tuple[list[dict], list[dict]]:
    root = repo / "out/eval_test/stage1"
    sources = [(root / "agent_sft/summary.csv", root / "agent_sft/trajectories.jsonl")]
    for train_seed in TRAIN_SEEDS:
        directory = root / winner / f"train_seed_{train_seed}"
        sources.append((directory / "summary.csv", directory / "trajectories.jsonl"))
    summaries: list[dict] = []
    trajectories: list[dict] = []
    for summary_path, trajectory_path in sources:
        summaries.extend(read_csv(summary_path))
        trajectories.extend(read_jsonl(trajectory_path))
    return summaries, trajectories


def audit_eval(
    summaries: list[dict], trajectories: list[dict], expected_items: int,
    algorithms: set[str], split: str,
) -> None:
    per_checkpoint_summary: dict[str, list[dict]] = defaultdict(list)
    per_checkpoint_trajectory: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        if algorithm_name(row["checkpoint"]) in algorithms:
            per_checkpoint_summary[row["checkpoint"]].append(row)
    for row in trajectories:
        if algorithm_name(row["checkpoint"]) in algorithms:
            per_checkpoint_trajectory[row["checkpoint"]].append(row)

    expected_checkpoints = 0
    for algorithm in algorithms:
        expected_checkpoints += 3 if algorithm in ALGORITHMS else 1
    if len(per_checkpoint_summary) != expected_checkpoints:
        raise RuntimeError(
            f"{split}: expected {expected_checkpoints} checkpoints, found "
            f"{sorted(per_checkpoint_summary)}"
        )
    if set(per_checkpoint_summary) != set(per_checkpoint_trajectory):
        raise RuntimeError(f"{split}: summary/trajectory checkpoint mismatch")

    for checkpoint, rows in per_checkpoint_summary.items():
        seeds = {int(row["seed"]) for row in rows}
        if seeds != set(DECODE_SEEDS) or len(rows) != len(DECODE_SEEDS):
            raise RuntimeError(f"{split}/{checkpoint}: decode seeds are {sorted(seeds)}")
        if any(int(row["num_trajectories"]) != expected_items for row in rows):
            raise RuntimeError(f"{split}/{checkpoint}: summary item count mismatch")
        trajectory_rows = per_checkpoint_trajectory[checkpoint]
        expected_rows = expected_items * len(DECODE_SEEDS)
        if len(trajectory_rows) != expected_rows:
            raise RuntimeError(
                f"{split}/{checkpoint}: expected {expected_rows} trajectories, "
                f"found {len(trajectory_rows)}"
            )
        keys = {(int(row["seed"]), int(row["dataset_index"])) for row in trajectory_rows}
        if len(keys) != expected_rows:
            raise RuntimeError(f"{split}/{checkpoint}: duplicate or missing trajectory key")
        if {index for _, index in keys} != set(range(expected_items)):
            raise RuntimeError(f"{split}/{checkpoint}: dataset indices are incomplete")


def aggregate_summaries(
    summaries: list[dict], algorithms: set[str], kl_references: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    checkpoint_rows: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        if algorithm_name(row["checkpoint"]) in algorithms:
            checkpoint_rows[row["checkpoint"]].append(row)

    per_checkpoint = []
    for checkpoint, rows in sorted(checkpoint_rows.items()):
        output = {
            "algorithm": algorithm_name(checkpoint),
            "checkpoint": checkpoint,
            "training_seed": checkpoint_train_seed(checkpoint) or "",
            "decode_seeds": len(rows),
            "kl_reference": kl_references[algorithm_name(checkpoint)],
        }
        for metric in METRICS:
            output[metric] = mean([float(row[metric]) for row in rows])
        per_checkpoint.append(output)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_checkpoint:
        grouped[row["algorithm"]].append(row)
    per_algorithm = []
    for algorithm, rows in sorted(grouped.items()):
        references = {row["kl_reference"] for row in rows}
        if len(references) != 1:
            raise RuntimeError(f"Mixed KL references for {algorithm}: {references}")
        output = {
            "algorithm": algorithm,
            "training_runs": len(rows),
            "kl_reference": references.pop(),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_training_seed_std"] = sample_std(values)
        per_algorithm.append(output)

    baseline = next(row for row in per_algorithm if row["algorithm"] == "agent_sft")
    for row in per_algorithm:
        for metric in METRICS:
            if metric == "kl_k3" and row["kl_reference"] != baseline["kl_reference"]:
                # KL values with different reference policies are not subtractable.
                row[f"{metric}_delta_vs_agent_sft"] = ""
            else:
                row[f"{metric}_delta_vs_agent_sft"] = (
                    row[f"{metric}_mean"] - baseline[f"{metric}_mean"]
                )
    return per_checkpoint, per_algorithm


def question_scores(trajectories: list[dict], algorithm: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in trajectories:
        if algorithm_name(row["checkpoint"]) == algorithm:
            grouped[int(row["dataset_index"])].append(float(row[metric]))
    if not grouped:
        raise RuntimeError(f"No trajectories for {algorithm}")
    return {index: mean(values) for index, values in grouped.items()}


def paired_bootstrap(
    trajectories: list[dict], algorithms: set[str], expected_items: int,
    samples: int, seed: int,
) -> list[dict]:
    baseline = question_scores(trajectories, "agent_sft", "task_accuracy")
    output = []
    for offset, algorithm in enumerate(sorted(algorithms - {"agent_sft"})):
        candidate = question_scores(trajectories, algorithm, "task_accuracy")
        if set(candidate) != set(baseline) or len(candidate) != expected_items:
            raise RuntimeError(f"Cannot pair {algorithm} with Agent-SFT")
        differences = np.asarray(
            [candidate[index] - baseline[index] for index in range(expected_items)],
            dtype=np.float64,
        )
        # Independent deterministic stream per algorithm keeps results stable if
        # the displayed algorithm set changes.
        local_rng = np.random.default_rng(seed + offset)
        bootstrap = np.empty(samples, dtype=np.float64)
        for start in range(0, samples, 256):
            size = min(256, samples - start)
            indices = local_rng.integers(0, expected_items, size=(size, expected_items))
            bootstrap[start:start + size] = differences[indices].mean(axis=1)
        output.append({
            "algorithm": algorithm,
            "baseline": "agent_sft",
            "items": expected_items,
            "bootstrap_samples": samples,
            "task_accuracy_delta": float(differences.mean()),
            "paired_item_bootstrap_ci95_low": quantile(bootstrap, 0.025),
            "paired_item_bootstrap_ci95_high": quantile(bootstrap, 0.975),
            "probability_delta_gt_zero": float((bootstrap > 0).mean()),
            "question_win_rate": float((differences > 0).mean()),
            "question_tie_rate": float((differences == 0).mean()),
            "question_loss_rate": float((differences < 0).mean()),
            "uncertainty_scope": "paired question sampling; training-seed std reported separately",
        })
    return output


def classify_failure(row: dict) -> str:
    if float(row["task_accuracy"]) == 1.0:
        return "success"
    if float(row["unfinished"]) == 1.0:
        return "unfinished"
    if float(row["format_valid_rate"]) == 0.0:
        return "format_invalid"
    if float(row["tool_call_valid_rate"]) == 0.0:
        return "tool_call_invalid"
    if float(row["tool_execution_success_rate"]) == 0.0:
        return "tool_execution_failed"
    if float(row["required_tool_coverage_rate"]) == 0.0:
        return "required_tool_missing"
    if float(row["answer_accuracy"]) == 0.0:
        return "answer_incorrect"
    if float(row["tool_evidence_coverage_rate"]) == 0.0:
        return "evidence_not_grounded"
    return "strict_verifier_rejection"


def failure_modes(trajectories: list[dict], algorithms: set[str], split: str) -> list[dict]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in trajectories:
        algorithm = algorithm_name(row["checkpoint"])
        if algorithm not in algorithms:
            continue
        counts[(algorithm, classify_failure(row))] += 1
        totals[algorithm] += 1
    rows = []
    for (algorithm, mode), count in sorted(counts.items()):
        rows.append({
            "split": split,
            "algorithm": algorithm,
            "failure_mode": mode,
            "count": count,
            "rate": count / totals[algorithm],
            "trajectories": totals[algorithm],
        })
    return rows


def audit_training(repo: Path) -> list[dict]:
    baseline_path = repo / "out/agent_sft_768.pth"
    baseline_state = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if not all(torch.isfinite(tensor).all() for tensor in baseline_state.values()):
        raise RuntimeError(f"Non-finite tensor in baseline checkpoint: {baseline_path}")
    output = []
    for algorithm in ALGORITHMS:
        for train_seed in TRAIN_SEEDS:
            path = repo / f"out/metrics/gsm8k_{algorithm}_full_s{train_seed}.jsonl"
            records = read_jsonl(path)
            final = records[-1]
            metrics = final["metrics"]
            if not metrics.get("budget_complete"):
                raise RuntimeError(f"Incomplete training budget: {path}")
            if metrics["candidate_groups"] != 6726 or metrics["candidate_trajectories"] != 53808:
                raise RuntimeError(f"Unexpected formal budget: {path}")
            detailed = next(
                row for row in reversed(records)
                if "wall_time_seconds" in row.get("metrics", {})
            )["metrics"]
            checkpoint = repo / (
                f"out/rl_full/{algorithm}/seed_{train_seed}/"
                f"gsm8k_{algorithm}_full_s{train_seed}_768.pth"
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.keys() != baseline_state.keys():
                raise RuntimeError(f"Checkpoint key mismatch: {checkpoint}")
            changed_tensors = 0
            changed_parameters = 0
            total_parameters = 0
            squared_change = 0.0
            squared_baseline = 0.0
            maximum_absolute_change = 0.0
            for key, baseline_tensor in baseline_state.items():
                candidate_tensor = state[key]
                if not torch.isfinite(candidate_tensor).all():
                    raise RuntimeError(f"Non-finite tensor {key}: {checkpoint}")
                baseline_float = baseline_tensor.float()
                difference = candidate_tensor.float() - baseline_float
                total_parameters += baseline_tensor.numel()
                nonzero = difference != 0
                if torch.any(nonzero):
                    changed_tensors += 1
                    changed_parameters += int(nonzero.sum())
                squared_change += float((difference * difference).sum())
                squared_baseline += float((baseline_float * baseline_float).sum())
                maximum_absolute_change = max(
                    maximum_absolute_change, float(difference.abs().max()),
                )
            output.append({
                "algorithm": algorithm,
                "training_seed": train_seed,
                "budget_complete": True,
                "budget_stop_reason": metrics["budget_stop_reason"],
                "candidate_groups": metrics["candidate_groups"],
                "candidate_trajectories": metrics["candidate_trajectories"],
                "effective_groups": metrics["effective_groups"],
                "effective_group_rate": metrics["effective_groups"] / metrics["candidate_groups"],
                "optimizer_updates": metrics["optimizer_updates"],
                "generated_tokens": metrics["generated_tokens"],
                "wall_time_seconds": detailed["wall_time_seconds"],
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256(checkpoint),
                "tensor_count": len(state),
                "nonfinite_tensors": 0,
                "changed_tensors_vs_agent_sft": changed_tensors,
                "changed_parameters_vs_agent_sft": changed_parameters,
                "total_parameters": total_parameters,
                "changed_parameter_fraction_vs_agent_sft": changed_parameters / total_parameters,
                "relative_l2_change_vs_agent_sft": math.sqrt(squared_change / squared_baseline),
                "maximum_absolute_change_vs_agent_sft": maximum_absolute_change,
            })
            del state
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inventory(paths: list[Path], repo: Path) -> list[dict]:
    return [
        {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(paths))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("out/stage1_final"))
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261005)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_summaries, validation_trajectories = collect_validation(repo)
    validation_algorithms = {"full_sft", "agent_sft", *ALGORITHMS}
    audit_eval(
        validation_summaries, validation_trajectories, 747,
        validation_algorithms, "validation",
    )
    checkpoint_rows, algorithm_rows = aggregate_summaries(
        validation_summaries,
        validation_algorithms,
        {
            "full_sft": "full_sft",
            "agent_sft": "full_sft",
            **{algorithm: "agent_sft" for algorithm in ALGORITHMS},
        },
    )
    winner = max(
        (row for row in algorithm_rows if row["algorithm"] in ALGORITHMS),
        key=lambda row: (row["task_accuracy_mean"], row["reward_mean"], row["answer_accuracy_mean"]),
    )["algorithm"]
    paired_validation = paired_bootstrap(
        validation_trajectories,
        {"agent_sft", *ALGORITHMS},
        747,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    training_rows = audit_training(repo)

    write_csv(output_dir / "training_runs.csv", training_rows)
    write_csv(output_dir / "validation_by_checkpoint.csv", checkpoint_rows)
    write_csv(output_dir / "validation_by_algorithm.csv", algorithm_rows)
    write_csv(output_dir / "validation_paired_bootstrap.csv", paired_validation)
    failure_rows = failure_modes(
        validation_trajectories, validation_algorithms, "validation",
    )

    test_algorithm_rows = None
    test_paired = None
    test_trajectory_count = 0
    if args.include_test:
        test_summaries, test_trajectories = collect_test(repo, winner)
        test_trajectory_count = len(test_trajectories)
        test_algorithms = {"agent_sft", winner}
        audit_eval(test_summaries, test_trajectories, 1319, test_algorithms, "test")
        test_checkpoint_rows, test_algorithm_rows = aggregate_summaries(
            test_summaries,
            test_algorithms,
            {algorithm: "agent_sft" for algorithm in test_algorithms},
        )
        test_paired = paired_bootstrap(
            test_trajectories,
            test_algorithms,
            1319,
            args.bootstrap_samples,
            args.bootstrap_seed + 100,
        )
        write_csv(output_dir / "test_by_checkpoint.csv", test_checkpoint_rows)
        write_csv(output_dir / "test_by_algorithm.csv", test_algorithm_rows)
        write_csv(output_dir / "test_paired_bootstrap.csv", test_paired)
        failure_rows.extend(failure_modes(test_trajectories, test_algorithms, "test"))
    write_csv(output_dir / "failure_modes.csv", failure_rows)

    evidence_inputs = [
        repo / "data/processed/gsm8k_agent/train_rl.jsonl",
        repo / "data/processed/gsm8k_agent/train_sft.jsonl",
        repo / "data/processed/gsm8k_agent/validation_rl.jsonl",
        repo / "data/processed/gsm8k_agent/test_rl.jsonl",
        repo / "dataset/manifests/gsm8k_agent.json",
        repo / "out/agent_sft_768.pth",
        repo / "model/model_minimind.py",
        repo / "dataset/lm_dataset.py",
        repo / "trainer/math_env.py",
        repo / "trainer/policy_optimization.py",
        repo / "trainer/rollout_engine.py",
        repo / "trainer/train_agent.py",
        repo / "scripts/eval_agent_rlvr.py",
        repo / "scripts/run_gsm8k_rl_full.sh",
        repo / "scripts/run_gsm8k_stage1_final_test.sh",
        repo / "scripts/analyze_gsm8k_stage1.py",
        repo / "docs/STAGE1_GSM8K_FINAL_REPORT.md",
    ]
    evidence_inputs.extend(
        repo / f"out/rl_full/{algorithm}/seed_{seed}/gsm8k_{algorithm}_full_s{seed}_768.pth"
        for algorithm in ALGORITHMS for seed in TRAIN_SEEDS
    )
    evidence = {
        "protocol": {
            "selection_split": "validation",
            "primary_metric": "strict task_accuracy",
            "tie_breakers": ["reward", "answer_accuracy"],
            "winner": winner,
            "test_used_for_selection": False,
            "training_seeds": list(TRAIN_SEEDS),
            "decode_seeds": list(DECODE_SEEDS),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "GSM8K question",
            "bootstrap_scope": "item-sampling uncertainty; does not replace training-seed variability",
        },
        "counts": {
            "train_rl": 6726,
            "train_sft": 6637,
            "validation": 747,
            "official_test": 1319,
            "formal_rl_checkpoints": 12,
            "validation_decode_trajectories": len(validation_trajectories),
            "test_decode_trajectories": test_trajectory_count,
        },
        "winner_validation": next(row for row in algorithm_rows if row["algorithm"] == winner),
        "winner_validation_paired": next(row for row in paired_validation if row["algorithm"] == winner),
        "test_complete": bool(args.include_test),
        "test_results": test_algorithm_rows,
        "test_paired": test_paired,
        "artifacts": hash_inventory(evidence_inputs, repo),
    }
    with (output_dir / "stage1_evidence.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    generated = [path for path in output_dir.iterdir() if path.is_file()]
    with (output_dir / "generated_artifacts.sha256").open("w", encoding="utf-8") as handle:
        for path in sorted(generated):
            if path.name == "generated_artifacts.sha256":
                continue
            handle.write(f"{sha256(path)}  {path.name}\n")

    print(json.dumps({
        "winner": winner,
        "validation": evidence["winner_validation"],
        "validation_paired": evidence["winner_validation_paired"],
        "test_complete": bool(args.include_test),
        "test": test_algorithm_rows,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
