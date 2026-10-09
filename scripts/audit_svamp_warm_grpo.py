"""Fail-closed readiness and result audit for SVAMP warm-start GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.gsm8k import sha256_file  # noqa: E402
from scripts.audit_qwen_stage2_results import (  # noqa: E402
    adapter_fingerprint,
    audit_evaluation,
    audit_grpo,
)
from trainer.math_env import canonical_number, safe_calculate  # noqa: E402
from trainer.qwen3_adapter import load_yaml_config  # noqa: E402


EXPECTED_LABELS = ["additional_sft_b_zero_shot", "svamp_grpo"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument(
        "--output", default="out/run_meta/qwen3_svamp_warm_grpo_audit.json"
    )
    return parser.parse_args()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_data() -> dict:
    manifest_path = ROOT / "dataset/manifests/svamp_agent.json"
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != "svamp_only_author_cv_folds_0_3_train_fold4_holdout":
        raise RuntimeError("unexpected SVAMP protocol")
    expected_counts = {"train_rl": 816, "holdout_rl": 184, "probe_rl": 128}
    ids = {}
    groups = {}
    report = {}
    for name, expected_count in expected_counts.items():
        metadata = manifest["files"][name]
        path = ROOT / metadata["path"]
        rows = read_jsonl(path)
        if len(rows) != expected_count or int(metadata["rows"]) != expected_count:
            raise RuntimeError(f"SVAMP {name} row-count mismatch")
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"SVAMP {name} fingerprint mismatch")
        ids[name] = {row["id"] for row in rows}
        groups[name] = {row["svamp_group_nums"] for row in rows}
        for row in rows:
            expression = row["svamp_infix_expression"]
            observed = safe_calculate(expression)["result"]
            if canonical_number(observed) != canonical_number(row["gt"][0]):
                raise RuntimeError(f"SVAMP oracle mismatch: {row['id']}")
            if row.get("required_tools") != ["calculate_math"]:
                raise RuntimeError(f"SVAMP required-tool mismatch: {row['id']}")
        report[name] = {
            "path": metadata["path"],
            "rows": len(rows),
            "sha256": metadata["sha256"],
        }
    if ids["train_rl"] & ids["holdout_rl"]:
        raise RuntimeError("SVAMP train/holdout ID leakage")
    if groups["train_rl"] & groups["holdout_rl"]:
        raise RuntimeError("SVAMP train/holdout variation-family leakage")
    if not ids["probe_rl"] <= ids["train_rl"]:
        raise RuntimeError("SVAMP probe is not a train subset")
    return report


def main():
    args = parse_args()
    grpo_config = load_yaml_config(
        ROOT / "configs/qwen3_4b/svamp/grpo_from_additional_sft_b.yaml"
    )
    eval_config = load_yaml_config(ROOT / "configs/qwen3_4b/svamp/eval_holdout.yaml")
    initial_adapter = grpo_config["initialization"]["adapter_path"]
    if eval_config["runs"][0]["adapter_path"] != initial_adapter:
        raise RuntimeError("zero-shot arm and GRPO initialization differ")
    if grpo_config["training"]["max_candidate_groups"] != 816:
        raise RuntimeError("SVAMP GRPO budget must cover all 816 train prompts")
    if grpo_config["training"]["reward_mode"] != "strict":
        raise RuntimeError("formal SVAMP GRPO must use strict reward")
    if grpo_config["rollout"] != {
        "num_generations": 8,
        "max_turns": 3,
        "max_new_tokens": 384,
        "max_total_length": 2500,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
    }:
        raise RuntimeError("unexpected SVAMP rollout protocol")
    report = {
        "status": "PASS",
        "data": audit_data(),
        "initial_adapter": adapter_fingerprint(initial_adapter),
        "result_required": args.require_results,
    }
    if args.require_results:
        labels = [run["label"] for run in eval_config["runs"]]
        if labels != EXPECTED_LABELS:
            raise RuntimeError(f"unexpected SVAMP evaluation labels: {labels}")
        report["grpo"] = audit_grpo(grpo_config, "svamp_grpo")
        report["holdout"] = audit_evaluation(
            eval_config,
            "test",
            required=True,
            expected_labels=EXPECTED_LABELS,
        )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SVAMP warm-start GRPO audit: PASS ({output_path})")


if __name__ == "__main__":
    main()
