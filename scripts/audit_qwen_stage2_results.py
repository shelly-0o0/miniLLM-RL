"""Fail closed unless the complete four-arm Stage 2 Track 1 result exists.

This is an artifact audit, not a model-quality test.  It proves that both
full GRPO branches consumed the configured training split, all adapters are
finite and fingerprinted, and frozen-split evaluation contains every arm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from dataset.gsm8k import sha256_file
from trainer.qwen3_adapter import load_yaml_config


EXPECTED_LABELS = ["base", "sft_only", "pure_grpo", "sft_grpo"]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit completed Qwen Stage 2 Track 1")
    parser.add_argument(
        "--require-test",
        action="store_true",
        help="Also require the full official-test four-arm evaluation.",
    )
    parser.add_argument(
        "--output",
        default="out/run_meta/qwen3_stage2_track1_results_audit.json",
    )
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def adapter_fingerprint(relative_path: str) -> dict:
    adapter_dir = ROOT / relative_path
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"incomplete adapter: {adapter_dir}")
    tensor_count = 0
    parameter_count = 0
    non_finite = []
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensor_count += 1
            parameter_count += tensor.numel()
            if not bool(torch.isfinite(tensor).all()):
                non_finite.append(name)
    if non_finite:
        raise RuntimeError(f"adapter contains non-finite tensors: {non_finite[:8]}")
    return {
        "path": relative_path,
        "adapter_model_sha256": sha256_file(model_path),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "non_finite_tensors": non_finite,
    }


def numeric_values(mapping: dict):
    for value in mapping.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield float(value)


def audit_sft(config: dict) -> dict:
    data_path = ROOT / config["data"]["train_path"]
    output = config["output"]
    final_path = ROOT / output["trainer_dir"] / "final_summary.json"
    final = read_json(final_path)
    rows = count_jsonl(data_path)
    if int(final["dataset_rows"]) != rows:
        raise RuntimeError(f"SFT row count mismatch: {final['dataset_rows']} != {rows}")
    train_metrics = final.get("train_metrics") or {}
    if not all(math.isfinite(value) for value in numeric_values(train_metrics)):
        raise RuntimeError("SFT summary contains non-finite metrics")
    return {
        "dataset_rows": rows,
        "data_sha256": sha256_file(data_path),
        "final_summary": str(final_path.relative_to(ROOT)),
        "train_metrics": train_metrics,
        "adapter": adapter_fingerprint(output["adapter_dir"]),
    }


def audit_grpo(config: dict, label: str) -> dict:
    data_path = ROOT / config["data"]["train_path"]
    dataset_rows = count_jsonl(data_path)
    output = config["output"]
    rows = read_jsonl(ROOT / output["metrics_path"])
    metrics = [
        row["metrics"] for row in rows
        if row.get("split") == "train" and "metrics" in row
    ]
    rejections = [
        row["metrics"] for row in rows
        if row.get("split") == "safety_rejection" and "metrics" in row
    ]
    finals = [row["final"] for row in rows if "final" in row]
    if not finals:
        raise RuntimeError(f"{label} has no terminal metrics record")
    final = finals[-1]
    group_size = int(config["rollout"]["num_generations"])
    expected_groups = dataset_rows * int(config["training"].get("epochs", 1))
    expected_trajectories = expected_groups * group_size
    if int(final["dataset_rows"]) != dataset_rows:
        raise RuntimeError(f"{label} final dataset row mismatch")
    if int(final["candidate_groups"]) != expected_groups:
        raise RuntimeError(
            f"{label} is partial: {final['candidate_groups']} / {expected_groups} groups"
        )
    if int(final["candidate_trajectories"]) != expected_trajectories:
        raise RuntimeError(f"{label} trajectory budget mismatch")
    observed_groups = {
        int(row["candidate_group"]) for row in metrics + rejections
    }
    missing = set(range(1, dataset_rows + 1)).difference(observed_groups)
    if missing:
        raise RuntimeError(f"{label} metrics omit {len(missing)} candidate groups")
    for index, row in enumerate(metrics):
        if not all(math.isfinite(value) for value in numeric_values(row)):
            raise RuntimeError(f"{label} has non-finite metric record {index}")
    for index, row in enumerate(rejections):
        if not all(math.isfinite(value) for value in numeric_values(row)):
            raise RuntimeError(f"{label} has non-finite rejection record {index}")
    max_mae = max(float(row["rollout_logprob_mae"]) for row in metrics)
    max_kl = max(float(row["kl_k3"]) for row in metrics)
    if max_mae > float(config["training"]["max_rollout_logprob_mae"]):
        raise RuntimeError(f"{label} rollout-logprob gate was exceeded")
    if max_kl > float(config["training"]["max_group_kl_k3"]):
        raise RuntimeError(f"{label} KL gate was exceeded")
    if len(rejections) != int(final.get("kl_rejections", 0)):
        raise RuntimeError(f"{label} KL rejection count mismatch")
    if len(rejections) > int(config["training"].get("max_kl_rejections", 0)):
        raise RuntimeError(f"{label} exceeded its KL rejection budget")
    for row in rejections:
        if float(row["group_kl_k3"]) <= float(row["threshold"]):
            raise RuntimeError(f"{label} rejected a group below the KL threshold")
    return {
        "dataset_rows": dataset_rows,
        "data_sha256": sha256_file(data_path),
        "group_size": group_size,
        "candidate_groups": expected_groups,
        "candidate_trajectories": expected_trajectories,
        "optimizer_updates": int(final["optimizer_updates"]),
        "metric_records": len(metrics),
        "kl_rejections": len(rejections),
        "max_rollout_logprob_mae": max_mae,
        "max_kl_k3": max_kl,
        "adapter": adapter_fingerprint(output["adapter_dir"]),
    }


def audit_evaluation(
    matrix: dict,
    split: str,
    *,
    required: bool,
    expected_labels: list[str] | None = None,
) -> dict | None:
    evaluation = matrix["evaluation"]
    output_dir = ROOT / evaluation["output_dir"] / split
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    if not summary_path.exists() and not required:
        return None
    summaries = read_json(summary_path)
    manifest = read_json(manifest_path)
    labels = [row["label"] for row in summaries]
    required_labels = expected_labels or EXPECTED_LABELS
    if labels != required_labels:
        raise RuntimeError(f"{split} evaluation arms mismatch: {labels}")
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"{split} evaluation manifest is not complete")
    data_path = ROOT / evaluation[f"{split}_path"]
    data_rows = count_jsonl(data_path)
    configured_limit = int(evaluation.get("limit", 0))
    prompts = data_rows if split == "test" else min(configured_limit or data_rows, data_rows)
    expected_trajectories = prompts * len(evaluation.get("decode_seeds", [42]))
    run_paths = {run["label"]: run.get("adapter_path") for run in matrix["runs"]}
    for row in summaries:
        if int(row["num_trajectories"]) != expected_trajectories:
            raise RuntimeError(f"{split}/{row['label']} trajectory count mismatch")
        if row.get("data_sha256") != sha256_file(data_path):
            raise RuntimeError(f"{split}/{row['label']} data fingerprint mismatch")
        adapter_path = run_paths[row["label"]]
        expected_hash = (
            adapter_fingerprint(adapter_path)["adapter_model_sha256"]
            if adapter_path else None
        )
        if row.get("adapter_model_sha256") != expected_hash:
            raise RuntimeError(f"{split}/{row['label']} adapter fingerprint mismatch")
        if not all(math.isfinite(value) for value in numeric_values(row)):
            raise RuntimeError(f"{split}/{row['label']} contains non-finite metrics")
    return {
        "summary": str(summary_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "data_rows": data_rows,
        "evaluated_prompts": prompts,
        "trajectories_per_arm": expected_trajectories,
        "results": summaries,
    }


def main():
    args = parse_args()
    config_dir = ROOT / "configs/qwen3_4b"
    sft_config = load_yaml_config(config_dir / "lora_sft_lm_head.yaml")
    pure_config = load_yaml_config(config_dir / "pure_grpo.yaml")
    warm_config = load_yaml_config(config_dir / "sft_grpo.yaml")
    matrix = load_yaml_config(config_dir / "eval_matrix.yaml")
    labels = [run["label"] for run in matrix["runs"]]
    if labels != EXPECTED_LABELS:
        raise RuntimeError(f"unexpected four-arm matrix: {labels}")
    report = {
        "status": "PASS",
        "sft_only": audit_sft(sft_config),
        "pure_grpo": audit_grpo(pure_config, "pure_grpo"),
        "sft_grpo": audit_grpo(warm_config, "sft_grpo"),
        "validation": audit_evaluation(matrix, "validation", required=True),
        "test": audit_evaluation(matrix, "test", required=args.require_test),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Stage 2 Track 1 result audit: PASS ({output_path})")


if __name__ == "__main__":
    main()
