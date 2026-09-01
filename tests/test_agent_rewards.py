import unittest

from scripts.eval_agent_rlvr import pack_trajectories
from trainer.train_agent import (
    TOOLS,
    calculate_rewards,
    checkpoint_state_dict,
    execute_tool,
    shift_action_mask,
    validate_gt_in_text,
)

import torch


class AgentRewardTest(unittest.TestCase):
    def test_rl_checkpoint_defaults_to_float32(self):
        model = torch.nn.Linear(3, 2)
        state = checkpoint_state_dict(model)
        self.assertTrue(all(value.device.type == "cpu" for value in state.values()))
        self.assertTrue(all(value.dtype == torch.float32 for value in state.values()))
        state16 = checkpoint_state_dict(model, "float16")
        self.assertTrue(all(value.dtype == torch.float16 for value in state16.values()))

    def test_calculator_uses_a_bounded_arithmetic_grammar(self):
        self.assertEqual(execute_tool("calculate_math", {"expression": "(2+3)*4"}), {"result": "20"})
        self.assertIsNone(execute_tool("calculate_math", {"expression": "().__class__.__mro__"}))
        self.assertIsNone(execute_tool("calculate_math", {"expression": "__import__('os').system('id')"}))
        self.assertIsNone(execute_tool("calculate_math", {"expression": "2**1000000"}))
        self.assertIsNone(execute_tool("calculate_math", {"expression": "1/0"}))

    def test_valid_tool_trajectory_is_verifiable(self):
        tool = [item for item in TOOLS if item["function"]["name"] == "calculate_math"]
        call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>'
        output = calculate_rewards(
            prompts=["question"],
            completions=["The result is 4."],
            gt_batch=[["4"]],
            tools_batch=[tool],
            num_gen=1,
            device="cpu",
            turn_outputs_batch=[[call, "The result is 4."]],
            unfinished_batch=[False],
            require_tool_call_for_success=True,
            return_details=True,
        )
        self.assertTrue(output.task_success.item())
        self.assertEqual(output.answer_accuracy.item(), 1.0)
        self.assertEqual(output.tool_call_valid.item(), 1.0)
        self.assertEqual(output.tool_execution_success.item(), 1.0)
        self.assertEqual(output.tool_evidence_coverage.item(), 1.0)

        strict = calculate_rewards(
            prompts=["question"], completions=["The result is 4."],
            gt_batch=[["4"]], tools_batch=[tool], num_gen=1, device="cpu",
            turn_outputs_batch=[[call, "The result is 4."]],
            unfinished_batch=[False], require_tool_call_for_success=True,
            reward_mode="strict", return_details=True,
        )
        self.assertEqual(strict.rewards.item(), 1.0)

    def test_format_bonus_cannot_fake_task_success(self):
        output = calculate_rewards(
            prompts=["question"],
            completions=["This is a well formatted but wrong answer."],
            gt_batch=[["4"]],
            tools_batch=[[]],
            num_gen=1,
            device="cpu",
            turn_outputs_batch=[["This is a well formatted but wrong answer."]],
            unfinished_batch=[False],
            return_details=True,
        )
        self.assertGreater(output.rewards.item(), -3.0)
        self.assertFalse(output.task_success.item())

    def test_distractor_tool_does_not_satisfy_required_tool(self):
        tools = [
            item for item in TOOLS
            if item["function"]["name"] in {"calculate_math", "get_current_time"}
        ]
        distractor = '<tool_call>{"name":"get_current_time","arguments":{}}</tool_call>'
        output = calculate_rewards(
            prompts=["question"], completions=["4"], gt_batch=[["4"]],
            tools_batch=[tools], num_gen=1,
            required_tools_batch=[["calculate_math"]], device="cpu",
            turn_outputs_batch=[[distractor, "4"]], unfinished_batch=[False],
            require_tool_call_for_success=True, reward_mode="strict",
            return_details=True,
        )
        self.assertEqual(output.required_tool_coverage.item(), 0.0)
        self.assertFalse(output.task_success.item())

    def test_wrong_execution_plus_guessed_answer_is_not_success(self):
        tool = [item for item in TOOLS if item["function"]["name"] == "calculate_math"]
        wrong_call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
        output = calculate_rewards(
            prompts=["compute 2+2"], completions=["4"], gt_batch=[["4"]],
            tools_batch=[tool], num_gen=1,
            required_tools_batch=[["calculate_math"]], device="cpu",
            turn_outputs_batch=[[wrong_call, "4"]], unfinished_batch=[False],
            require_tool_call_for_success=True, reward_mode="strict",
            return_details=True,
        )
        self.assertEqual(output.answer_accuracy.item(), 1.0)
        self.assertEqual(output.tool_evidence_coverage.item(), 0.0)
        self.assertFalse(output.task_success.item())

    def test_structured_evidence_uses_all_tool_results_not_only_last_line(self):
        evidence = '{"temperature":"20°C"}\n{"datetime":"2025-03-07 14:30:00"}'
        gt = ["20°C", "2025-03-07 14:30:00"]
        self.assertEqual(
            validate_gt_in_text(evidence, gt, final_answer_only=False), set(gt)
        )
        self.assertNotEqual(validate_gt_in_text(evidence, gt), set(gt))

    def test_multi_tool_success_requires_both_execution_results(self):
        tools = [
            item for item in TOOLS
            if item["function"]["name"] in {"get_current_weather", "get_current_time"}
        ]
        weather = '<tool_call>{"name":"get_current_weather","arguments":{"location":"南京"}}</tool_call>'
        clock = '<tool_call>{"name":"get_current_time","arguments":{"timezone":"Asia/Shanghai"}}</tool_call>'
        output = calculate_rewards(
            prompts=["query"], completions=["温度20°C，时间2025-03-07 14:30:00"],
            gt_batch=[["20°C", "2025-03-07 14:30:00"]], tools_batch=[tools],
            num_gen=1, required_tools_batch=[["get_current_weather", "get_current_time"]],
            device="cpu", turn_outputs_batch=[[weather + clock, "温度20°C，时间2025-03-07 14:30:00"]],
            unfinished_batch=[False], require_tool_call_for_success=True,
            reward_mode="strict", return_details=True,
        )
        self.assertEqual(output.required_tool_coverage.item(), 1.0)
        self.assertEqual(output.tool_evidence_coverage.item(), 1.0)
        self.assertTrue(output.task_success.item())

    def test_shaped_curriculum_rewards_complete_grounded_tool_use(self):
        tools = [
            item for item in TOOLS
            if item["function"]["name"] in {"get_current_weather", "get_current_time"}
        ]
        weather = '<tool_call>{"name":"get_current_weather","arguments":{"location":"南京"}}</tool_call>'
        clock = '<tool_call>{"name":"get_current_time","arguments":{"timezone":"Asia/Shanghai"}}</tool_call>'
        common = dict(
            prompts=["query"],
            gt_batch=[["20°C", "2025-03-07 14:30:00"]],
            tools_batch=[tools],
            num_gen=1,
            required_tools_batch=[["get_current_weather", "get_current_time"]],
            device="cpu",
            unfinished_batch=[False],
            require_tool_call_for_success=True,
            reward_mode="shaped",
            return_details=True,
        )
        complete = calculate_rewards(
            completions=["温度20°C，时间2025-03-07 14:30:00"],
            turn_outputs_batch=[[weather + clock, "温度20°C，时间2025-03-07 14:30:00"]],
            **common,
        )
        partial = calculate_rewards(
            completions=["温度20°C"],
            turn_outputs_batch=[[weather, "温度20°C"]],
            **common,
        )
        guessed = calculate_rewards(
            completions=["温度20°C，时间2025-03-07 14:30:00"],
            turn_outputs_batch=[["温度20°C，时间2025-03-07 14:30:00"]],
            **common,
        )
        self.assertGreater(complete.rewards.item(), partial.rewards.item())
        self.assertGreater(partial.rewards.item(), guessed.rewards.item())
        self.assertEqual(complete.tool_evidence_coverage.item(), 1.0)
        self.assertEqual(partial.required_tool_coverage.item(), 0.5)
        self.assertFalse(guessed.task_success.item())

    def test_unfinished_trajectory_never_succeeds(self):
        output = calculate_rewards(
            prompts=["question"],
            completions=["4"],
            gt_batch=[["4"]],
            tools_batch=[[]],
            num_gen=1,
            device="cpu",
            turn_outputs_batch=[["4"]],
            unfinished_batch=[True],
            return_details=True,
        )
        self.assertFalse(output.task_success.item())

    def test_numeric_verifier_rejects_substring_reward_hack(self):
        output = calculate_rewards(
            prompts=["question"],
            completions=["The answer is 14, not the requested value."],
            gt_batch=[["4"]],
            tools_batch=[[]],
            num_gen=1,
            device="cpu",
            turn_outputs_batch=[["The answer is 14, not the requested value."]],
            unfinished_batch=[False],
            return_details=True,
        )
        self.assertEqual(output.answer_accuracy.item(), 0.0)
        self.assertFalse(output.task_success.item())

        wrong_final = calculate_rewards(
            prompts=["question"],
            completions=["I considered 4, but final answer: 14"],
            gt_batch=[["4"]], tools_batch=[[]], num_gen=1, device="cpu",
            turn_outputs_batch=[["I considered 4, but final answer: 14"]],
            unfinished_batch=[False], reward_mode="strict", return_details=True,
        )
        self.assertEqual(wrong_final.rewards.item(), -1.0)

    def test_post_rollout_left_truncation_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds --max_total_len"):
            pack_trajectories(
                tokenizer=object(),
                prompt_ids=[[1, 2, 3]],
                response_ids=[[4, 5]],
                response_masks=[[1, 1]],
                max_total_len=4,
                device="cpu",
            )

    def test_multi_turn_action_mask_keeps_actions_after_each_eos(self):
        # prompt, assistant turn 1 (including EOS), observation, assistant
        # turn 2 (including EOS), padding
        full_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 0]], dtype=torch.float32)
        shifted = shift_action_mask(full_mask)
        self.assertEqual(shifted.tolist(), [[0, 1, 1, 0, 0, 1, 1, 0]])
        self.assertEqual(int(shifted.sum().item()), 4)


if __name__ == "__main__":
    unittest.main()
