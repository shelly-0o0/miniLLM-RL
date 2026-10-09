import unittest
from pathlib import Path

from scripts.prepare.prepare_svamp_agent import (
    convert_row,
    prefix_to_infix,
    render_number_tokens,
)
from trainer.math_env import safe_calculate
from trainer.qwen3_adapter import load_yaml_config


class SVAMPWarmGRPOTests(unittest.TestCase):
    def test_prefix_equation_is_replayed_by_shared_calculator(self):
        expression = prefix_to_infix(
            "+ * number0 number1 number2", ["3", "4", "5"]
        )
        self.assertEqual(safe_calculate(expression)["result"], "17")

    def test_number_token_rendering_does_not_confuse_indices(self):
        numbers = [str(index) for index in range(11)]
        rendered = render_number_tokens("number10 minus number1", numbers)
        self.assertEqual(rendered, "10 minus 1")

    def test_svamp_conversion_builds_verified_agent_rl_row(self):
        row = {
            "Numbers": "76 25",
            "Body": "each pack costs number0 dollars .",
            "Ques": "what is the price after a number1 dollar discount ?",
            "Equation": "- number0 number1",
            "Answer": "51.0",
            "group_nums": "[1, 2]",
            "Type": "Subtraction",
        }
        converted = convert_row(row, fold=4, row_index=0)
        self.assertEqual(converted["gt"], ["51"])
        self.assertEqual(converted["required_tools"], ["calculate_math"])
        self.assertEqual(converted["oracle_observations"], [{"result": "51"}])
        self.assertEqual(converted["source_split"], "derived_holdout")

    def test_svamp_grpo_uses_additional_sft_b_and_full_softmax(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "configs/qwen3_4b/svamp"
        grpo = load_yaml_config(directory / "grpo_from_additional_sft_b.yaml")
        probe = load_yaml_config(directory / "pre_grpo_probe.yaml")
        evaluation = load_yaml_config(directory / "eval_holdout.yaml")
        start = "out/stage2_track2/qwen3_4b/additional_sft_b_s42_adapter"
        self.assertEqual(grpo["initialization"]["adapter_path"], start)
        self.assertEqual(probe["initialization"]["adapter_path"], start)
        self.assertEqual(evaluation["runs"][0]["adapter_path"], start)
        self.assertEqual(grpo["data"]["train_path"], "data/processed/svamp_agent/train_rl.jsonl")
        self.assertEqual(grpo["training"]["max_candidate_groups"], 816)
        self.assertEqual(grpo["training"]["reward_mode"], "strict")
        self.assertEqual(
            (
                grpo["rollout"]["temperature"],
                grpo["rollout"]["top_k"],
                grpo["rollout"]["top_p"],
            ),
            (1.0, 0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
