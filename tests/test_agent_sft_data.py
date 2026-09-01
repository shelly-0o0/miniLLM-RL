import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "prepare_agent_sft_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_agent_sft_data", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AgentSFTDataTests(unittest.TestCase):
    def test_balanced_limit_is_deterministic_and_category_balanced(self):
        rows = [
            {"id": f"{category}-{index}", "rlvr_category": category}
            for category in ("a", "b", "c") for index in range(5)
        ]
        selected = MODULE.balanced_limit(rows, limit=6, seed=7)
        repeated = MODULE.balanced_limit(rows, limit=6, seed=7)
        self.assertEqual(selected, repeated)
        counts = {category: 0 for category in ("a", "b", "c")}
        for row in selected:
            counts[row["rlvr_category"]] += 1
        self.assertEqual(counts, {"a": 2, "b": 2, "c": 2})

    def test_extracts_multiple_arithmetic_calls_and_verifies_answers(self):
        row = {
            "conversations": [
                {"role": "system", "content": "", "tools": "[]"},
                {"role": "user", "content": "计算3**3，同时请计算(418+250)*10，并计算9570/22"},
                {"role": "assistant", "content": ""},
            ],
            "gt": ["27", "6680", "435"],
        }
        converted = MODULE.from_math(row)
        self.assertIsNotNone(converted)
        calls = converted["conversations"][-5]["tool_calls"]
        self.assertIn("calculate_math", calls)
        self.assertIn("9570/22", calls)

    def test_rejects_expression_answer_mismatch(self):
        row = {
            "conversations": [
                {"role": "system", "content": "", "tools": "[]"},
                {"role": "user", "content": "Compute (2+3)*4"},
                {"role": "assistant", "content": ""},
            ],
            "gt": ["999"],
        }
        self.assertIsNone(MODULE.from_math(row))

    def test_verified_tool_requires_paired_oracles(self):
        row = {
            "conversations": [
                {"role": "system", "content": "", "tools": "[]"},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": ""},
            ],
            "gt": ["x"],
            "oracle_calls": [{"name": "f", "arguments": {}}],
            "oracle_observations": [],
        }
        self.assertIsNone(MODULE.from_verified_tool(row))


if __name__ == "__main__":
    unittest.main()
