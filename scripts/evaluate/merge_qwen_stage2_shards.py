"""Validate and atomically merge per-arm Qwen Stage 2 evaluation shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.gsm8k import sha256_file  # noqa: E402
from trainer.qwen3_adapter import load_yaml_config, require_section  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/qwen3_4b/eval_matrix.yaml")
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Match an evaluate_qwen_stage2.py --limit override used by every "
            "shard. Use 0 for the full split."
        ),
    )
    return parser.parse_args()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def count_and_validate_trajectories(
    path: Path,
    *,
    label: str,
    expected_count: int,
    expected_indices: set[int],
    expected_seeds: set[int],
) -> None:
    observed_pairs: set[tuple[int, int]] = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if row.get("label") != label:
                raise RuntimeError(f"trajectory label mismatch at {path}:{line_number}")
            pair = (int(row["seed"]), int(row["index"]))
            if pair in observed_pairs:
                raise RuntimeError(f"duplicate trajectory key {pair} in {path}")
            observed_pairs.add(pair)
            count += 1
    expected_pairs = {
        (seed, index) for seed in expected_seeds for index in expected_indices
    }
    if count != expected_count or observed_pairs != expected_pairs:
        raise RuntimeError(
            f"trajectory coverage mismatch for {label}: rows={count}, "
            f"unique_keys={len(observed_pairs)}, expected={expected_count}"
        )


def numeric_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from numeric_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from numeric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    evaluation = require_section(config, "evaluation")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("eval config requires a non-empty runs list")
    labels = [run["label"] for run in runs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate matrix labels: {labels}")

    data_path = ROOT / evaluation[f"{args.split}_path"]
    data_sha256 = sha256_file(data_path)
    row_count = sum(1 for line in data_path.open(encoding="utf-8") if line.strip())
    limit = int(evaluation.get("limit", 0)) if args.limit is None else args.limit
    prompt_count = row_count if not limit else min(row_count, limit)
    seeds = [int(seed) for seed in evaluation.get("decode_seeds", [42])]
    expected_trajectories = prompt_count * len(seeds)
    expected_indices = set(range(prompt_count))
    expected_seeds = set(seeds)
    sampling = {
        key: evaluation.get(key)
        for key in (
            "max_prompt_length", "max_turns", "max_new_tokens",
            "temperature", "top_k", "top_p",
        )
    }

    base_dir = ROOT / evaluation["output_dir"]
    shard_root = base_dir / f"{args.split}_shards"
    final_dir = base_dir / args.split
    if final_dir.exists():
        raise FileExistsError(
            f"canonical evaluation output already exists: {final_dir}; "
            "archive it before merging"
        )

    config_sha256 = sha256_file(config_path)
    summaries = []
    source_manifests = []
    validated = []
    for run in runs:
        label = run["label"]
        shard_dir = shard_root / label
        manifest = read_json(shard_dir / "manifest.json")
        shard_summaries = read_json(shard_dir / "summary.json")
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete evaluation shard: {label}")
        if manifest.get("config_sha256") != config_sha256:
            raise RuntimeError(f"config fingerprint mismatch in shard: {label}")
        if manifest.get("data_sha256") != data_sha256:
            raise RuntimeError(f"data fingerprint mismatch in shard: {label}")
        if manifest.get("matrix_labels") != labels:
            raise RuntimeError(f"matrix label mismatch in shard: {label}")
        if manifest.get("selected_labels") != [label]:
            raise RuntimeError(f"selected label mismatch in shard: {label}")
        if manifest.get("decode_seeds") != seeds:
            raise RuntimeError(f"decode seed mismatch in shard: {label}")
        if manifest.get("sampling") != sampling:
            raise RuntimeError(f"sampling mismatch in shard: {label}")
        if int(manifest.get("limit", -1)) != limit:
            raise RuntimeError(f"limit mismatch in shard: {label}")
        if len(shard_summaries) != 1 or shard_summaries[0].get("label") != label:
            raise RuntimeError(f"summary label mismatch in shard: {label}")
        summary = shard_summaries[0]
        if int(summary.get("num_trajectories", -1)) != expected_trajectories:
            raise RuntimeError(f"trajectory count mismatch in shard: {label}")
        if summary.get("data_sha256") != data_sha256:
            raise RuntimeError(f"summary data fingerprint mismatch: {label}")
        if not all(math.isfinite(number) for number in numeric_values(summary)):
            raise RuntimeError(f"non-finite summary metric in shard: {label}")

        adapter_path = run.get("adapter_path")
        expected_adapter_hash = None
        if adapter_path:
            expected_adapter_hash = sha256_file(
                ROOT / adapter_path / "adapter_model.safetensors"
            )
        if summary.get("adapter_model_sha256") != expected_adapter_hash:
            raise RuntimeError(f"adapter fingerprint mismatch in shard: {label}")
        trajectory_path = shard_dir / f"{label}_trajectories.jsonl"
        count_and_validate_trajectories(
            trajectory_path,
            label=label,
            expected_count=expected_trajectories,
            expected_indices=expected_indices,
            expected_seeds=expected_seeds,
        )
        summaries.append(summary)
        source_manifests.append(str((shard_dir / "manifest.json").relative_to(ROOT)))
        validated.append((label, trajectory_path, expected_adapter_hash, adapter_path))

    temporary = base_dir / f".{args.split}.merge-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(config_path, temporary / "used_config.yaml")
        merged_runs = []
        for label, source, adapter_hash, adapter_path in validated:
            destination = temporary / f"{label}_trajectories.jsonl"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            merged_runs.append({
                "label": label,
                "adapter_path": adapter_path,
                "adapter_model_sha256": adapter_hash,
                "trajectory_path": str(
                    (final_dir / destination.name).relative_to(ROOT)
                ),
                "trajectory_sha256": sha256_file(destination),
                "num_trajectories": expected_trajectories,
            })
        (temporary / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "split": args.split,
            "data_path": str(data_path.relative_to(ROOT)),
            "data_sha256": data_sha256,
            "limit": limit,
            "decode_seeds": seeds,
            "sampling": sampling,
            "require_all_runs": bool(evaluation.get("require_all_runs", False)),
            "matrix_labels": labels,
            "selected_labels": labels,
            "sharded": True,
            "merge_source_manifests": source_manifests,
            "runs": merged_runs,
            "summary_path": str((final_dir / "summary.json").relative_to(ROOT)),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "COMPLETE",
        "output_dir": str(final_dir.relative_to(ROOT)),
        "labels": labels,
        "trajectories_per_arm": expected_trajectories,
        "data_sha256": data_sha256,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
