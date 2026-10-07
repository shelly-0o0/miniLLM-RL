import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dataset.qwen_stage2 import AssistantOnlyCollator
from trainer.agent_chat import (
    audit_agent_generation_prefixes,
    build_assistant_only_example,
    normalize_agent_messages,
    render_agent_chat,
    resolve_agent_open_thinking,
)
from trainer.policy_optimization import compute_policy_loss, positive_kl_estimate
from trainer.qwen3_adapter import load_yaml_config, selected_action_logps
from trainer.qwen_grpo_objective import trajectory_grpo_loss
from trainer.train_qwen_grpo import (
    group_k3_diagnostics,
    resolve_kl_gate_policy,
    train_without_dropout,
    validate_on_policy_sampling,
    write_kl_failure_diagnostic,
)
from trainer.train_qwen_lora_sft import (
    snapshot_trainable_parameters,
    summarize_trainable_change,
    weighted_causal_lm_loss,
)


class FakeToolTokenizer:
    chat_template = "fake"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        chunks = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            calls = message.get("tool_calls") or []
            call_text = json.dumps(calls, sort_keys=True) if calls else ""
            chunks.append(f"<{role}>{content}{call_text}</{role}>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        text = "".join(chunks)
        return [ord(character) for character in text] if tokenize else text


class ContextSensitiveQwenLikeTokenizer:
    """Mimic Qwen3's generation prompt changing a completed tool-call prefix."""

    chat_template = "fake-qwen <|im_start|> <tool_call> enable_thinking"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        chunks = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            calls = message.get("tool_calls") or []
            if calls:
                content = "".join(
                    f"<tool_call>{json.dumps(call, sort_keys=True)}</tool_call>"
                    for call in calls
                )
            if role == "assistant" and (
                index == len(messages) - 1
                or (index > 0 and messages[index - 1].get("role") == "tool")
            ):
                content = f"<think>\n\n</think>\n\n{content}"
            chunks.append(
                f"<|im_start|>{role}\n{content}<|im_end|>\n"
            )
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
            if kwargs.get("enable_thinking") is False:
                chunks.append("<think>\n\n</think>\n\n")
        text = "".join(chunks)
        return [ord(character) for character in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        output = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return output


class IndexedLogitModel(torch.nn.Module):
    def __init__(self, vocab_size=13):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 5)
        self.projection = torch.nn.Linear(5, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        logits = self.projection(self.embedding(input_ids))
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep, :]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class QwenStage2Tests(unittest.TestCase):
    def test_structure_weighted_sft_loss_emphasizes_protocol_tokens(self):
        logits = torch.zeros(1, 3, 5, requires_grad=True)
        labels = torch.tensor([[-100, 1, 3]])
        ordinary = weighted_causal_lm_loss(
            logits,
            labels,
            structure_token_ids={3},
            structure_token_weight=1.0,
        )
        weighted = weighted_causal_lm_loss(
            logits,
            labels,
            structure_token_ids={3},
            structure_token_weight=8.0,
        )
        self.assertAlmostEqual(ordinary.item(), weighted.item(), places=6)

        improved = logits.detach().clone()
        improved[0, 1, 3] = 2.0
        improved.requires_grad_(True)
        ordinary_improvement = ordinary - weighted_causal_lm_loss(
            improved,
            labels,
            structure_token_ids={3},
            structure_token_weight=1.0,
        )
        weighted_improvement = weighted - weighted_causal_lm_loss(
            improved,
            labels,
            structure_token_ids={3},
            structure_token_weight=8.0,
        )
        self.assertGreater(weighted_improvement.item(), ordinary_improvement.item())
        weighted_improvement.backward()
        self.assertTrue(torch.isfinite(improved.grad).all())

    def test_qwen_rollout_prompt_matches_sft_turn_boundaries(self):
        tokenizer = ContextSensitiveQwenLikeTokenizer()
        initial = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "QUESTION"},
        ]
        first_mode = resolve_agent_open_thinking(
            tokenizer, initial, requested_open_thinking=False
        )
        self.assertTrue(first_mode)
        first_prompt = render_agent_chat(
            tokenizer, initial, tools=[], tokenize=False,
            add_generation_prompt=True, open_thinking=first_mode,
        )
        self.assertTrue(first_prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", first_prompt)

        after_tool = initial + [
            {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
            {"role": "tool", "content": "OBS"},
        ]
        final_mode = resolve_agent_open_thinking(
            tokenizer, after_tool, requested_open_thinking=False
        )
        self.assertFalse(final_mode)
        final_prompt = render_agent_chat(
            tokenizer, after_tool, tools=[], tokenize=False,
            add_generation_prompt=True, open_thinking=final_mode,
        )
        self.assertTrue(final_prompt.endswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ))
        complete = after_tool + [
            {"role": "assistant", "content": "FINAL"},
        ]
        audit = audit_agent_generation_prefixes(tokenizer, complete)
        self.assertEqual(audit["assistant_turns"], 2)
        self.assertEqual(
            [turn["open_thinking"] for turn in audit["turns"]],
            [True, False],
        )

    def test_sft_smoke_update_summary_detects_real_parameter_change(self):
        model = torch.nn.Linear(2, 2)
        before = snapshot_trainable_parameters(model)
        with torch.no_grad():
            model.weight.add_(0.25)
        summary = summarize_trainable_change(model, before)
        self.assertEqual(summary["tensor_count"], 2)
        self.assertEqual(summary["changed_tensors"], 1)
        self.assertAlmostEqual(summary["maximum_absolute_change"], 0.25)
        self.assertEqual(summary["non_finite_tensors"], [])

    def test_grpo_train_mode_disables_dropout_without_disabling_gradients(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Dropout(p=0.5), torch.nn.Linear(3, 2)
        )
        train_without_dropout(model)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        model(torch.ones(1, 3)).sum().backward()
        self.assertIsNotNone(model[0].weight.grad)

    def test_kl_safety_failure_writes_token_level_diagnostic(self):
        class Tokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=False):
                return f"token-{ids[0]}"

        packed = {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "action_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            write_kl_failure_diagnostic(
                path,
                tokenizer=Tokenizer(),
                candidate_group=74,
                trajectory_index=3,
                packed=packed,
                current_logps=torch.tensor([-2.0, -10.0]),
                old_logps=torch.tensor([-2.0, -10.0]),
                reference_logps=torch.tensor([-2.0, 0.0]),
                completion="answer",
                turns=["answer"],
                trace={"stop_reason": "final_answer"},
                reward=1.0,
                advantage=0.5,
                threshold=10.0,
            )
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["candidate_group"], 74)
        self.assertEqual(report["trajectory_index"], 3)
        self.assertEqual(report["outlier_tokens"][0]["token_id"], 13)
        self.assertEqual(report["outlier_tokens"][0]["token_text"], "token-13")
        self.assertGreater(report["mean_kl_k3"], 10.0)
        self.assertEqual(report["max_abs_current_minus_old"], 0.0)

    def test_grpo_rejects_sampling_distribution_that_differs_from_logprobs(self):
        validate_on_policy_sampling(
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
        )
        for bad in (
            {"temperature": 0.7, "top_k": 0, "top_p": 1.0},
            {"temperature": 1.0, "top_k": 20, "top_p": 1.0},
            {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_on_policy_sampling(bad)

    def test_group_kl_gate_matches_token_global_objective_not_worst_trajectory(self):
        policy = [torch.zeros(384) for _ in range(8)]
        reference = [torch.zeros(384) for _ in range(8)]
        reference[0][0] = 9.365
        group_mean, trajectory_means, token_count = group_k3_diagnostics(
            policy, reference
        )
        self.assertEqual(token_count, 8 * 384)
        self.assertGreater(trajectory_means[0].item(), 10.0)
        self.assertLess(group_mean.item(), 10.0)
        expected = sum(
            positive_kl_estimate(current, ref).sum()
            for current, ref in zip(policy, reference)
        ) / token_count
        self.assertTrue(torch.allclose(group_mean, expected))

    def test_kl_gate_group_rejection_is_bounded(self):
        self.assertEqual(resolve_kl_gate_policy({}), ("raise", 0, 0))
        self.assertEqual(
            resolve_kl_gate_policy({
                "kl_gate_action": "reject_group",
                "max_kl_rejections": 64,
                "max_consecutive_kl_rejections": 3,
            }),
            ("reject_group", 64, 3),
        )
        with self.assertRaises(ValueError):
            resolve_kl_gate_policy({"kl_gate_action": "reject_group"})
        with self.assertRaises(ValueError):
            resolve_kl_gate_policy({"kl_gate_action": "ignore"})
        with self.assertRaises(ValueError):
            resolve_kl_gate_policy({
                "kl_gate_action": "reject_group",
                "max_kl_rejections": 1,
                "max_consecutive_kl_rejections": 1,
                "gradient_accumulation_steps": 2,
            })

    def test_normalizes_tools_and_json_tool_calls(self):
        conversations = [
            {"role": "system", "content": "s", "tools": '[{"name":"x"}]'},
            {"role": "assistant", "content": "", "tool_calls": '[{"name":"x","arguments":{}}]'},
        ]
        messages, tools = normalize_agent_messages(conversations)
        self.assertEqual(tools, [{"name": "x"}])
        self.assertNotIn("tools", messages[0])
        self.assertIsInstance(messages[1]["tool_calls"], list)

    def test_assistant_only_mask_excludes_user_and_tool_tokens(self):
        tokenizer = FakeToolTokenizer()
        conversations = [
            {"role": "system", "content": "SYS", "tools": "[]"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "CALL"},
            {"role": "tool", "content": "OBSERVATION"},
            {"role": "assistant", "content": "FINAL"},
        ]
        row = build_assistant_only_example(tokenizer, conversations, max_length=4096)
        supervised = "".join(
            chr(token) for token, label in zip(row["input_ids"], row["labels"])
            if label != -100
        )
        self.assertIn("CALL", supervised)
        self.assertIn("FINAL", supervised)
        self.assertNotIn("QUESTION", supervised)
        self.assertNotIn("OBSERVATION", supervised)

    def test_context_sensitive_qwen_mask_uses_completed_role_blocks(self):
        tokenizer = ContextSensitiveQwenLikeTokenizer()
        conversations = [
            {"role": "system", "content": "SYS", "tools": "[]"},
            {"role": "user", "content": "QUESTION"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "calculate_math", "arguments": {}}],
            },
            {"role": "tool", "content": "OBSERVATION"},
            {"role": "assistant", "content": "FINAL"},
        ]
        row = build_assistant_only_example(tokenizer, conversations, max_length=4096)
        supervised = "".join(
            chr(token) for token, label in zip(row["input_ids"], row["labels"])
            if label != -100
        )
        self.assertIn("calculate_math", supervised)
        self.assertIn("FINAL", supervised)
        self.assertNotIn("QUESTION", supervised)
        self.assertNotIn("OBSERVATION", supervised)

    def test_collator_uses_distinct_padding_values(self):
        collator = AssistantOnlyCollator(pad_token_id=9, pad_to_multiple_of=4)
        batch = collator([
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([-100, 2])},
            {"input_ids": torch.tensor([3]), "attention_mask": torch.tensor([1]), "labels": torch.tensor([3])},
        ])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["input_ids"][1, 1:].tolist(), [9, 9, 9])
        self.assertEqual(batch["attention_mask"][1, 1:].tolist(), [0, 0, 0])
        self.assertEqual(batch["labels"][1, 1:].tolist(), [-100, -100, -100])

    def test_selected_action_logps_use_only_requested_positions(self):
        torch.manual_seed(3)
        model = IndexedLogitModel()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        action_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        selected = selected_action_logps(
            model, ids, torch.ones_like(ids), action_mask, blocked_token_ids=[0]
        )
        full = model(ids).logits.float().clone()
        full[..., 0] = -torch.inf
        expected = torch.log_softmax(full[:, [1, 3], :], -1).gather(
            -1, ids[:, [2, 4]].unsqueeze(-1)
        ).squeeze(-1)[0]
        self.assertTrue(torch.allclose(selected, expected, atol=1e-6))
        selected.sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)

    def test_streaming_grpo_matches_dense_objective(self):
        torch.manual_seed(5)
        group_size, width = 4, 6
        current = torch.randn(group_size, width, requires_grad=True)
        old = torch.randn(group_size, width)
        reference = torch.randn(group_size, width)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.float32)
        advantages = torch.tensor([-1.0, -0.2, 0.3, 1.1])
        dense = compute_policy_loss(
            loss_type="grpo", current_logps=current, old_logps=old,
            reference_logps=reference, advantages=advantages,
            completion_mask=mask, beta=0.02, grpo_epsilon=0.2,
        )
        total_tokens = int(mask.sum().item())
        streamed = current.sum() * 0
        for index in range(group_size):
            keep = mask[index].bool()
            streamed = streamed + trajectory_grpo_loss(
                current[index, keep], old[index, keep], reference[index, keep],
                advantage=advantages[index], group_size=group_size,
                total_action_tokens=total_tokens, beta=0.02, epsilon=0.2,
            ).loss
        self.assertTrue(torch.allclose(streamed, dense.loss, atol=1e-6))

    def test_configs_encode_the_four_run_matrix(self):
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(root / "configs/qwen3_4b/eval_matrix.yaml")
        self.assertEqual(
            [run["label"] for run in config["runs"]],
            ["base", "sft_only", "pure_grpo", "sft_grpo"],
        )
        self.assertEqual(config["model"]["name"], "Qwen/Qwen3-4B-Base")
        self.assertTrue(config["evaluation"]["require_all_runs"])

        directory = root / "configs/qwen3_4b"
        repair = load_yaml_config(directory / "lora_sft_lm_head.yaml")
        pure = load_yaml_config(directory / "pure_grpo.yaml")
        warm = load_yaml_config(directory / "sft_grpo.yaml")
        probe = load_yaml_config(directory / "pre_grpo_probe.yaml")
        expected_targets = {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj", "lm_head",
        }
        for value in (repair, pure, warm, probe, config):
            self.assertEqual(set(value["lora"]["target_modules"]), expected_targets)
        repaired_path = "out/stage2/qwen3_4b/sft_lm_head_w8_s42_adapter"
        self.assertEqual(warm["initialization"]["adapter_path"], repaired_path)
        self.assertEqual(probe["initialization"]["adapter_path"], repaired_path)
        self.assertEqual(config["runs"][1]["adapter_path"], repaired_path)
        self.assertEqual(repair["training"]["structure_token_weight"], 8.0)
        self.assertEqual(repair["training"]["max_steps"], 0)
        self.assertEqual(pure["training"]["reward_mode"], "shaped")
        self.assertEqual(warm["training"]["reward_mode"], "shaped")
        for value in (pure, warm):
            self.assertEqual(value["rollout"]["temperature"], 1.0)
            self.assertEqual(value["rollout"]["top_k"], 0)
            self.assertEqual(value["rollout"]["top_p"], 1.0)
            self.assertEqual(value["training"]["kl_gate_action"], "reject_group")
            self.assertGreater(value["training"]["max_kl_rejections"], 0)
            self.assertGreater(
                value["training"]["max_consecutive_kl_rejections"], 0
            )
        for section in (probe["probe"], config["evaluation"]):
            self.assertEqual(section["temperature"], 1.0)
            self.assertEqual(section["top_k"], 0)
            self.assertEqual(section["top_p"], 1.0)
        self.assertEqual(
            pure["training"]["policy_device"],
            pure["training"]["reference_device"],
        )
        self.assertEqual(
            warm["training"]["policy_device"],
            warm["training"]["reference_device"],
        )


if __name__ == "__main__":
    unittest.main()
