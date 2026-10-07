import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_qwen_track2_results import (
    exact_mcnemar_pvalue,
    paired_bootstrap_interval,
    wilson_interval,
)
from scripts.evaluate.probe_qwen_track2 import aggregate_probe_groups
from scripts.prepare.prepare_gsm8k_track2 import build_track2_rows, split_ab
from trainer.qwen3_adapter import load_yaml_config


def source_row(index, annotated=True, split="train"):
    calculation = f"<<{index}+1={index + 1}>>" if annotated else ""
    return {
        "id": f"gsm8k_{split}_{index:05d}",
        "question": f"question {split} {index}",
        "answer": f"work {calculation}\n#### {index + 1}",
        "gold_answer": str(index + 1),
        "original_split": split,
    }


class QwenTrack2Tests(unittest.TestCase):
    def test_official_test_statistics_are_paired_and_finite(self):
        low, high = wilson_interval(25, 100)
        self.assertLess(low, 0.25)
        self.assertGreater(high, 0.25)
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_pvalue(1, 0), 1.0)
        interval = paired_bootstrap_interval(
            np.asarray([True, False, True]),
            np.asarray([True, False, True]),
            samples=100,
            seed=42,
        )
        self.assertEqual(interval, [0.0, 0.0])

    def test_ab_split_is_balanced_deterministic_and_disjoint(self):
        rows = [source_row(index) for index in range(11)]
        first = split_ab(rows, seed=42)
        second = split_ab(rows, seed=42)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (5, 6))
        self.assertFalse(
            {row["id"] for row in first[0]} & {row["id"] for row in first[1]}
        )
        self.assertEqual(
            {row["id"] for row in first[0] + first[1]},
            {row["id"] for row in rows},
        )

    def test_track2_rl_keeps_all_rows_and_sft_filters_missing_oracles(self):
        train = [source_row(index, annotated=index != 3) for index in range(8)]
        test = [source_row(100, split="test"), source_row(101, split="test")]
        output = build_track2_rows(
            train, test, split_seed=2, probe_seed=7, probe_size=1
        )
        self.assertEqual(len(output["a_rl"]) + len(output["b_rl"]), 8)
        self.assertEqual(len(output["a_sft"]) + len(output["b_sft"]), 7)
        self.assertEqual(len(output["test_rl"]), 2)
        a_ids = {row["id"] for row in output["a_rl"]}
        b_ids = {row["id"] for row in output["b_rl"]}
        self.assertFalse(a_ids & b_ids)
        self.assertTrue({row["id"] for row in output["a_sft"]} <= a_ids)
        self.assertTrue({row["id"] for row in output["b_sft"]} <= b_ids)
        self.assertEqual(len(output["a_compare_rl"]), len(output["b_compare_rl"]))
        self.assertTrue(
            {row["id"] for row in output["a_compare_rl"]}
            <= {row["id"] for row in output["a_sft"]}
        )
        self.assertTrue(
            {row["id"] for row in output["b_compare_rl"]}
            <= {row["id"] for row in output["b_sft"]}
        )

    def test_probe_aggregation_distinguishes_reachability_and_effective_groups(self):
        groups = [
            {
                "num_trajectories": 4, "success_count": 1,
                "pass_at_1": 0, "pass_at_k": 1, "effective_group": 1,
                "zero_variance_group": 0, "reward_std": 0.8,
                "shaped_reward_mean": 0.2, "shaped_reward_std": 0.4,
                "shaped_nonzero_variance_group": 1,
                "answer_accuracy": 0.25, "format_valid_rate": 1,
                "tool_call_valid_rate": 0.5, "tool_execution_success_rate": 0.5,
                "required_tool_coverage_rate": 0.5, "tool_evidence_coverage_rate": 0.25,
                "protocol_progress": 1.0,
                "unfinished_rate": 0, "avg_response_tokens": 100,
            },
            {
                "num_trajectories": 4, "success_count": 0,
                "pass_at_1": 0, "pass_at_k": 0, "effective_group": 0,
                "zero_variance_group": 1, "reward_std": 0,
                "shaped_reward_mean": -1.0, "shaped_reward_std": 0,
                "shaped_nonzero_variance_group": 0,
                "answer_accuracy": 0, "format_valid_rate": 1,
                "tool_call_valid_rate": 0, "tool_execution_success_rate": 0,
                "required_tool_coverage_rate": 0, "tool_evidence_coverage_rate": 0,
                "protocol_progress": 0.5,
                "unfinished_rate": 0.25, "avg_response_tokens": 120,
            },
        ]
        summary = aggregate_probe_groups(groups)
        self.assertEqual(summary["pass_at_k"], 0.5)
        self.assertEqual(summary["trajectory_task_accuracy"], 0.125)
        self.assertEqual(summary["effective_group_rate"], 0.5)
        self.assertEqual(summary["zero_variance_group_rate"], 0.5)
        self.assertEqual(summary["shaped_nonzero_variance_group_rate"], 0.5)

    def test_track2_branches_share_agent_sft_a_start(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "configs/qwen3_4b/track2"
        configs = {
            name: load_yaml_config(directory / f"{name}.yaml")
            for name in ("additional_sft_b", "grpo_a", "grpo_b", "pre_grpo_probe")
        }
        starts = {
            value["initialization"]["adapter_path"] for value in configs.values()
        }
        self.assertEqual(
            starts,
            {
                "out/stage2_track2/qwen3_4b/"
                "agent_sft_a_lm_head_w8_s42_adapter"
            },
        )
        self.assertEqual(configs["grpo_a"]["rollout"], configs["grpo_b"]["rollout"])
        self.assertEqual(configs["grpo_a"]["rollout"]["num_generations"], 8)
        self.assertEqual(configs["grpo_a"]["training"]["reward_mode"], "strict")
        self.assertEqual(configs["grpo_b"]["training"]["reward_mode"], "strict")
        for name in ("grpo_a", "grpo_b"):
            config = configs[name]
            self.assertEqual(config["training"]["kl_gate_action"], "reject_group")
            self.assertGreater(config["training"]["max_kl_rejections"], 0)


if __name__ == "__main__":
    unittest.main()
