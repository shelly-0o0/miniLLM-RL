import json
import tempfile
import unittest
from pathlib import Path

from dataset.gsm8k import assert_disjoint, build_manifest, extract_gold_answer, normalize_rows, split_train_validation, write_jsonl
from trainer.math_env import extract_final_answer, parse_tool_calls, safe_calculate, verify_answer
from scripts.prepare.prepare_gsm8k_agent_data import make_rl_row, make_sft_row, replay_annotations


class GSM8KPipelineTest(unittest.TestCase):
    def test_gold_extraction_and_deterministic_split(self):
        rows = normalize_rows(
            [{"question": f"question {i}", "answer": f"reasoning\n#### {i:,}"} for i in range(20)],
            "train",
        )
        first = split_train_validation(rows, 0.2, 42)
        second = split_train_validation(rows, 0.2, 42)
        self.assertEqual(first, second)
        self.assertEqual(extract_gold_answer("work\n#### 1,234"), "1234")
        self.assertEqual((len(first[0]), len(first[1])), (16, 4))

    def test_leakage_is_rejected(self):
        row = {"id": "a", "question": "Same question"}
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint({"train": [row], "test": [{"id": "b", "question": " same   QUESTION "}]})

    def test_manifest_contains_count_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            write_jsonl(path, [{"id": "one"}, {"id": "two"}])
            manifest = build_manifest({"train": path}, 42)
            self.assertEqual(manifest["splits"]["train"]["rows"], 2)
            self.assertEqual(len(manifest["splits"]["train"]["sha256"]), 64)

    def test_shared_math_environment(self):
        self.assertEqual(safe_calculate({"expression": "(2+3)*4"}), {"result": "20"})
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('id')")
        self.assertEqual(parse_tool_calls('<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>')[0]["name"], "calculate_math")
        self.assertEqual(extract_final_answer("reason 4; final answer: 14"), "14")
        self.assertTrue(verify_answer("The answer is 1/2", "0.5"))
        self.assertFalse(verify_answer("I considered 4, but final answer: 14", "4"))

    def test_agent_conversion_replays_oracles_and_hides_gold_from_prompt(self):
        row = {
            "id": "gsm8k_train_00000", "question": "John has 2 bags with 3 apples each.",
            "answer": "He has <<2*3=6>> apples.\n#### 6", "gold_answer": "6", "split": "train",
        }
        calls, observations = replay_annotations(row["answer"])
        self.assertEqual(calls[0]["arguments"]["expression"], "2*3")
        self.assertEqual(observations, [{"result": "6"}])
        rl = make_rl_row(row)
        prompt_text = json.dumps(rl["conversations"][:-1])
        self.assertNotIn("Final answer: 6", prompt_text)
        self.assertEqual(rl["gt"], ["6"])
        sft = make_sft_row(row)
        self.assertEqual(sft["conversations"][-1]["content"], "Final answer: 6")

    def test_agent_conversion_rejects_bad_gsm8k_annotation(self):
        with self.assertRaisesRegex(ValueError, "oracle mismatch"):
            replay_annotations("bad annotation <<2+2=5>>\n#### 5")


if __name__ == "__main__":
    unittest.main()
