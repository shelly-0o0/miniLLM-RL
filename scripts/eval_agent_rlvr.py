"""Fixed-set evaluation for MiniMind RLVR / Agentic-RL checkpoints.

The script produces per-trajectory JSONL evidence and a summary CSV.  It uses
the same deterministic verifiers and multi-turn environment as
``trainer/train_agent.py`` while keeping evaluation prompts and seeds fixed.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import AgentRLDataset
from model.model_minimind import MiniMindConfig
from trainer.policy_optimization import positive_kl_estimate
from trainer.rollout_engine import TorchRolloutEngine, compute_per_token_logps
from trainer.train_agent import calculate_rewards, rollout_batch, shift_action_mask
from trainer.trainer_utils import init_model, setup_seed


def pack_trajectories(tokenizer, prompt_ids, response_ids, response_masks, max_total_len, device):
    trajectory_lengths = [len(prompt) + len(response) for prompt, response in zip(prompt_ids, response_ids)]
    max_trajectory_len = max(trajectory_lengths, default=0)
    if max_trajectory_len > max_total_len:
        raise RuntimeError(
            "An evaluation trajectory exceeds --max_total_len "
            f"({max_trajectory_len} > {max_total_len}). Left truncation changes the model context "
            "used for KL measurement. Increase --max_total_len or reduce --max_turns / "
            "--max_gen_len and rerun the complete fixed-set evaluation."
        )
    samples = []
    for prompt, response, response_mask in zip(prompt_ids, response_ids, response_masks):
        ids = prompt + response
        mask = [0] * len(prompt) + response_mask
        samples.append((ids, mask))
    max_len = max(len(ids) for ids, _ in samples)
    input_ids = torch.tensor([
        ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _ in samples
    ], device=device)
    full_response_mask = torch.tensor([
        mask + [0] * (max_len - len(mask)) for _, mask in samples
    ], device=device, dtype=torch.float32)
    response_mask = shift_action_mask(full_response_mask)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask, response_mask


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_checkpoint(args, weight, seed, dataset, reference_model, tokenizer):
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        max_position_embeddings=args.max_position_embeddings,
    )
    model, _ = init_model(
        config, weight, tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir, device=args.device,
    )
    model.eval()
    engine = TorchRolloutEngine(model, tokenizer, args.device, autocast_ctx=None)
    setup_seed(seed)
    totals = {
        "reward": [], "task_accuracy": [], "answer_accuracy": [],
        "format_valid_rate": [], "tool_call_valid_rate": [],
        "tool_execution_success_rate": [], "response_length": [],
        "required_tool_coverage_rate": [],
        "tool_evidence_coverage_rate": [],
        "unfinished": [], "kl_k3": [],
    }
    trajectory_records = []
    generated_tokens = 0
    started = time.time()

    for batch_start in range(0, len(dataset), args.batch_size):
        examples = [dataset[index] for index in range(batch_start, min(batch_start + args.batch_size, len(dataset)))]
        messages = [item["messages"] for item in examples]
        tools = [item["tools"] for item in examples]
        required_tools = [item["required_tools"] for item in examples]
        gt = [item["gt"] for item in examples]
        prompts = [
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, tools=item_tools)
            for item, item_tools in zip(messages, tools)
        ]
        with torch.no_grad():
            rollout_values = rollout_batch(
                engine, tokenizer, messages, tools, args.num_generations,
                max_turns=args.max_turns, max_new_tokens=args.max_gen_len,
                thinking_ratio=args.thinking_ratio, device=args.device,
            )
            (completions, _, prompt_ids, response_ids, response_masks, _,
             turn_outputs, unfinished) = rollout_values
            reward_output = calculate_rewards(
                prompts, completions, gt, tools, args.num_generations,
                required_tools, reward_model=None, device=args.device,
                turn_outputs_batch=turn_outputs,
                unfinished_batch=unfinished,
                require_tool_call_for_success=bool(args.require_tool_call_for_success),
                reward_mode=args.reward_mode,
                return_details=True,
            )
            input_ids, attention_mask, completion_mask = pack_trajectories(
                tokenizer, prompt_ids, response_ids, response_masks,
                args.max_total_len, args.device,
            )
            current_logps = compute_per_token_logps(
                model, input_ids, input_ids.size(1) - 1, attention_mask=attention_mask
            ).float()
            reference_logps = compute_per_token_logps(
                reference_model, input_ids, input_ids.size(1) - 1,
                attention_mask=attention_mask,
            ).float()
            per_token_kl = positive_kl_estimate(current_logps, reference_logps)
            row_kl = (per_token_kl * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)

        lengths = completion_mask.sum(1).cpu().tolist()
        generated_tokens += int(sum(lengths))
        fields = {
            "reward": reward_output.rewards.float().cpu().tolist(),
            "task_accuracy": reward_output.task_success.float().cpu().tolist(),
            "answer_accuracy": reward_output.answer_accuracy.float().cpu().tolist(),
            "format_valid_rate": reward_output.format_valid.float().cpu().tolist(),
            "tool_call_valid_rate": reward_output.tool_call_valid.float().cpu().tolist(),
            "tool_execution_success_rate": reward_output.tool_execution_success.float().cpu().tolist(),
            "required_tool_coverage_rate": reward_output.required_tool_coverage.float().cpu().tolist(),
            "tool_evidence_coverage_rate": reward_output.tool_evidence_coverage.float().cpu().tolist(),
            "response_length": lengths,
            "unfinished": [float(value) for value in unfinished],
            "kl_k3": row_kl.cpu().tolist(),
        }
        for key, values in fields.items():
            totals[key].extend(values)
        for local_index, completion in enumerate(completions):
            prompt_offset = local_index // args.num_generations
            trajectory_records.append({
                "checkpoint": weight,
                "seed": seed,
                "dataset_index": batch_start + prompt_offset,
                "generation_index": local_index % args.num_generations,
                "gt": gt[prompt_offset],
                "completion": completion,
                **{key: fields[key][local_index] for key in fields},
            })

    elapsed = max(time.time() - started, 1e-9)
    tensors = {key: torch.tensor(values, dtype=torch.float32) for key, values in totals.items()}
    grouped_success = tensors["task_accuracy"].view(-1, args.num_generations)
    success_per_group = grouped_success.sum(dim=1)
    summary = {
        "checkpoint": weight,
        "seed": seed,
        "num_trajectories": len(trajectory_records),
        **{key: value.mean().item() for key, value in tensors.items()},
        "reward_std": tensors["reward"].std(unbiased=False).item(),
        "zero_variance_success_group_rate": (
            (success_per_group == 0) | (success_per_group == args.num_generations)
        ).float().mean().item(),
        "dapo_effective_group_rate": (
            (success_per_group > 0) & (success_per_group < args.num_generations)
        ).float().mean().item(),
        "p95_response_length": torch.quantile(tensors["response_length"], 0.95).item(),
        "tokens_per_second": generated_tokens / elapsed,
    }
    del engine, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, trajectory_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind RLVR checkpoints on a fixed Agent dataset")
    parser.add_argument("--weights", required=True, help="逗号分隔，例如 grpo,cispo,dapo,gspo")
    parser.add_argument("--reference_weight", default="full_sft", help="KL参考策略")
    parser.add_argument("--data_path", default="./dataset/agent_rl_math.jsonl")
    parser.add_argument("--save_dir", default="./out")
    parser.add_argument("--tokenizer_path", default="./model")
    parser.add_argument("--output_dir", default="./out/eval_rlvr")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_gen_len", type=int, default=768)
    parser.add_argument("--max_total_len", type=int, default=2500)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--thinking_ratio", type=float, default=0.0)
    parser.add_argument("--require_tool_call_for_success", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reward_mode", choices=["strict", "shaped"], default="strict")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trajectory_path = os.path.join(args.output_dir, "trajectories.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    if os.path.exists(trajectory_path) or os.path.exists(summary_path):
        raise SystemExit(
            f"Refusing to mix/overwrite an existing evaluation in {args.output_dir}; "
            "choose a new --output_dir or archive the old directory explicitly"
        )
    if args.limit < 1 or args.batch_size < 1 or args.num_generations < 1:
        parser.error("limit, batch_size and num_generations must be >= 1")

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        max_position_embeddings=args.max_position_embeddings,
    )
    reference_model, tokenizer = init_model(
        config, args.reference_weight, tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir, device=args.device,
    )
    reference_model.eval().requires_grad_(False)
    full_dataset = AgentRLDataset(args.data_path, tokenizer, max_length=args.max_position_embeddings)
    # A prefix is intentional: every checkpoint sees exactly the same persisted
    # dataset order.  Shuffle the JSONL once externally if a random holdout is desired.
    full_dataset.samples = full_dataset.samples.select(range(min(args.limit, len(full_dataset)))) if hasattr(full_dataset.samples, "select") else full_dataset.samples[:args.limit]

    summaries = []
    for weight in [item.strip() for item in args.weights.split(",") if item.strip()]:
        for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
            summary, trajectories = evaluate_checkpoint(
                args, weight, seed, full_dataset, reference_model, tokenizer
            )
            summaries.append(summary)
            append_jsonl(trajectory_path, trajectories)
            print(json.dumps(summary, ensure_ascii=False))

    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {summary_path} and {trajectory_path}")


if __name__ == "__main__":
    main()
