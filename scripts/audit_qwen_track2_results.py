"""Fail closed unless the complete five-arm Stage 2 Track 2 result exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_qwen_stage2_results import (  # noqa: E402
    audit_evaluation,
    audit_grpo,
    audit_sft,
)
from trainer.qwen3_adapter import load_yaml_config  # noqa: E402


EXPECTED_LABELS = [
    "base",
    "agent_sft_a",
    "grpo_a_seen",
    "grpo_b_new_data",
    "additional_sft_b",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="out/run_meta/qwen3_stage2_track2_results_audit.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = ROOT / "configs/qwen3_4b/track2"
    sft_a = load_yaml_config(config_dir / "agent_sft_a_lm_head.yaml")
    additional_sft_b = load_yaml_config(config_dir / "additional_sft_b.yaml")
    grpo_a = load_yaml_config(config_dir / "grpo_a.yaml")
    grpo_b = load_yaml_config(config_dir / "grpo_b.yaml")
    matrix = load_yaml_config(config_dir / "eval_test.yaml")
    labels = [run["label"] for run in matrix["runs"]]
    if labels != EXPECTED_LABELS:
        raise RuntimeError(f"unexpected five-arm Track 2 matrix: {labels}")
    if grpo_a["training"]["max_candidate_groups"] != grpo_b["training"]["max_candidate_groups"]:
        raise RuntimeError("Track 2 GRPO A/B candidate budgets differ")
    if grpo_a["rollout"] != grpo_b["rollout"]:
        raise RuntimeError("Track 2 GRPO A/B rollout settings differ")
    report = {
        "status": "PASS",
        "agent_sft_a": audit_sft(sft_a),
        "additional_sft_b": audit_sft(additional_sft_b),
        "grpo_a": audit_grpo(grpo_a, "grpo_a"),
        "grpo_b": audit_grpo(grpo_b, "grpo_b"),
        "test": audit_evaluation(
            matrix,
            "test",
            required=True,
            expected_labels=EXPECTED_LABELS,
        ),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Stage 2 Track 2 result audit: PASS ({output_path})")


if __name__ == "__main__":
    main()
