"""Pre-GRPO reachability probe for one or more Stage 2 data pools.

The probe never updates model parameters.  It samples one G-sized trajectory
group per prompt from the shared Agent-SFT(A) checkpoint and measures whether
strictly correct, tool-grounded trajectories are already reachable on seen A
and new-to-SFT B prompts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.gsm8k import sha256_file
from dataset.lm_dataset import AgentRLDataset
from trainer.agent_chat import render_agent_chat
from trainer.qwen3_adapter import (
    load_policy_model,
    load_stage2_tokenizer,
    load_yaml_config,
    require_section,
)
from trainer.rollout_engine import TorchRolloutEngine
from trainer.train_agent import calculate_rewards, rollout_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 pre-GRPO reachability probe")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/qwen3_4b/track2/pre_grpo_probe.yaml"),
    )
    parser.add_argument(
        "--pool",
        default="all",
        help="Configured pool name, or 'all' for every configured pool.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit prompts per pool; 0 means the full configured probe file.",
    )
    parser.add_argument(
        "--num-generations", type=int, default=None,
        help="Override G for a cheap engineering smoke; formal probe uses config G=8.",
    )
    parser.add_argument(
        "--output-tag", default=None,
        help="Write to a tagged subdirectory so smoke results cannot replace the formal probe.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Override the configured Agent-SFT(A) adapter for a smoke chain.",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Probe the base model with a fresh zero-equivalent LoRA adapter.",
    )
    return parser.parse_args()


def average(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def aggregate_probe_groups(groups: list[dict]) -> dict:
    """Aggregate prompt-group diagnostics independently of model execution."""

    if not groups:
        raise ValueError("probe produced no groups")
    return {
        "num_prompts": len(groups),
        "num_trajectories": sum(row["num_trajectories"] for row in groups),
        "pass_at_1": average(groups, "pass_at_1"),
        "pass_at_k": average(groups, "pass_at_k"),
        "trajectory_task_accuracy": (
            sum(row["success_count"] for row in groups)
            / sum(row["num_trajectories"] for row in groups)
        ),
        "effective_group_rate": average(groups, "effective_group"),
        "zero_variance_group_rate": average(groups, "zero_variance_group"),
        "mean_group_reward_std": average(groups, "reward_std"),
        "mean_shaped_reward": average(groups, "shaped_reward_mean"),
        "mean_shaped_reward_std": average(groups, "shaped_reward_std"),
        "shaped_nonzero_variance_group_rate": average(
            groups, "shaped_nonzero_variance_group"
        ),
        "answer_accuracy": average(groups, "answer_accuracy"),
        "format_valid_rate": average(groups, "format_valid_rate"),
        "tool_call_valid_rate": average(groups, "tool_call_valid_rate"),
        "tool_execution_success_rate": average(
            groups, "tool_execution_success_rate"
        ),
        "required_tool_coverage_rate": average(
            groups, "required_tool_coverage_rate"
        ),
        "tool_evidence_coverage_rate": average(
            groups, "tool_evidence_coverage_rate"
        ),
        "protocol_progress": average(groups, "protocol_progress"),
        "unfinished_rate": average(groups, "unfinished_rate"),
        "avg_response_tokens": average(groups, "avg_response_tokens"),
    }


def main():
    cli = parse_args()
    config_path = Path(cli.config).resolve()
    config = load_yaml_config(config_path)
    model_config = require_section(config, "model")
    lora_config = require_section(config, "lora")
    initialization = require_section(config, "initialization")
    probe = require_section(config, "probe")
    if not torch.cuda.is_available():
        raise RuntimeError("Track 2 probe requires CUDA")

    if cli.base_model and cli.adapter_path:
        raise ValueError("--base-model and --adapter-path are mutually exclusive")
    configured_adapter = (
        None
        if cli.base_model
        else (cli.adapter_path or initialization.get("adapter_path"))
    )
    adapter_path = (ROOT / configured_adapter) if configured_adapter else None
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"missing Stage 2 adapter: {adapter_path}")
    output_dir = ROOT / probe["output_dir"]
    if cli.output_tag:
        if not cli.output_tag.replace("-", "").replace("_", "").isalnum():
            raise ValueError("--output-tag contains unsupported characters")
        output_dir = output_dir.with_name(f"{output_dir.name}_{cli.output_tag}")
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"probe output already exists: {output_dir}; choose a new --output-tag"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "used_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "runtime_overrides.json").write_text(
        json.dumps(
            {
                "pool": cli.pool,
                "limit": cli.limit,
                "num_generations": cli.num_generations,
                "output_tag": cli.output_tag,
                "adapter_path_override": cli.adapter_path,
                "base_model": cli.base_model,
                "resolved_adapter_path": (
                    str(adapter_path.relative_to(ROOT)) if adapter_path else None
                ),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    device = probe.get("device", "cuda:0")
    tokenizer = load_stage2_tokenizer(
        str(adapter_path) if adapter_path else model_config["name"],
        cache_dir=model_config.get("cache_dir"),
        padding_side="left",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_im_end_as_eos=bool(model_config.get("use_im_end_as_eos", True)),
    )
    model = load_policy_model(
        model_config,
        lora_config,
        device=device,
        adapter_path=str(adapter_path) if adapter_path else None,
        trainable=False,
    ).eval()
    engine = TorchRolloutEngine(
        model,
        tokenizer,
        device=device,
        autocast_ctx=torch.autocast(
            device_type="cuda",
            dtype=getattr(torch, model_config.get("compute_dtype", "bfloat16")),
        ),
    )

    group_size = int(cli.num_generations or probe.get("num_generations", 8))
    if group_size < 2:
        raise ValueError("probe requires at least two generations per prompt")
    if cli.pool != "all" and cli.pool not in probe["pools"]:
        raise ValueError(
            f"unknown probe pool {cli.pool!r}; expected one of "
            f"{sorted(probe['pools'])} or 'all'"
        )
    selected_pools = list(probe["pools"]) if cli.pool == "all" else [cli.pool]
    all_summaries = {}
    seed = int(config.get("seed", 42))

    for pool_name in selected_pools:
        data_path = ROOT / probe["pools"][pool_name]
        dataset = AgentRLDataset(
            data_path,
            tokenizer,
            max_length=int(probe.get("max_prompt_length", 1024)),
        )
        limit = len(dataset) if cli.limit is None or cli.limit == 0 else min(cli.limit, len(dataset))
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        groups, trajectories = [], []
        started = time.time()
        for index in range(limit):
            item = dataset[index]
            values = rollout_batch(
                engine,
                tokenizer,
                [item["messages"]],
                [item["tools"]],
                group_size,
                max_turns=int(probe.get("max_turns", 3)),
                max_new_tokens=int(probe.get("max_new_tokens", 384)),
                thinking_ratio=0.0,
                temperature=float(probe.get("temperature", 0.7)),
                top_k=int(probe.get("top_k", 20)),
                top_p=float(probe.get("top_p", 0.95)),
                device=device,
            )
            completions, _, _, _, response_masks, _, turns, unfinished, traces = values
            prompt_text = render_agent_chat(
                tokenizer,
                item["messages"],
                tools=item["tools"],
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
            reward = calculate_rewards(
                [prompt_text],
                completions,
                [item["gt"]],
                [item["tools"]],
                group_size,
                [item["required_tools"]],
                device=device,
                turn_outputs_batch=turns,
                unfinished_batch=unfinished,
                require_tool_call_for_success=True,
                reward_mode="strict",
                return_details=True,
            )
            shaped_reward = calculate_rewards(
                [prompt_text],
                completions,
                [item["gt"]],
                [item["tools"]],
                group_size,
                [item["required_tools"]],
                device=device,
                turn_outputs_batch=turns,
                unfinished_batch=unfinished,
                require_tool_call_for_success=True,
                reward_mode="shaped",
                return_details=True,
            )
            successes = reward.task_success.bool()
            success_count = int(successes.sum().item())
            reward_std = reward.rewards.float().std(unbiased=False).item()
            shaped_reward_std = (
                shaped_reward.rewards.float().std(unbiased=False).item()
            )
            group = {
                "pool": pool_name,
                "index": index,
                "id": dataset.samples[index].get("id"),
                "num_trajectories": group_size,
                "success_count": success_count,
                "pass_at_1": float(successes[0].item()),
                "pass_at_k": float(success_count > 0),
                "effective_group": float(0 < success_count < group_size),
                "zero_variance_group": float(reward_std < 1e-6),
                "reward_std": reward_std,
                "shaped_reward_mean": shaped_reward.rewards.mean().item(),
                "shaped_reward_std": shaped_reward_std,
                "shaped_nonzero_variance_group": float(
                    shaped_reward_std >= 1e-6
                ),
                "answer_accuracy": reward.answer_accuracy.mean().item(),
                "format_valid_rate": reward.format_valid.mean().item(),
                "tool_call_valid_rate": reward.tool_call_valid.mean().item(),
                "tool_execution_success_rate": reward.tool_execution_success.mean().item(),
                "required_tool_coverage_rate": reward.required_tool_coverage.mean().item(),
                "tool_evidence_coverage_rate": reward.tool_evidence_coverage.mean().item(),
                "protocol_progress": shaped_reward.protocol_progress.mean().item(),
                "unfinished_rate": sum(bool(value) for value in unfinished) / group_size,
                "avg_response_tokens": sum(map(sum, response_masks)) / group_size,
            }
            groups.append(group)
            for trajectory_index in range(group_size):
                trajectories.append({
                    "pool": pool_name,
                    "index": index,
                    "id": group["id"],
                    "trajectory_index": trajectory_index,
                    "completion": completions[trajectory_index],
                    "turn_outputs": turns[trajectory_index],
                    "trace": traces[trajectory_index],
                    "reward": reward.rewards[trajectory_index].item(),
                    "shaped_reward": shaped_reward.rewards[trajectory_index].item(),
                    "task_success": bool(successes[trajectory_index]),
                    "answer_accuracy": reward.answer_accuracy[trajectory_index].item(),
                    "tool_evidence_coverage": reward.tool_evidence_coverage[trajectory_index].item(),
                    "protocol_progress": shaped_reward.protocol_progress[trajectory_index].item(),
                    "unfinished": bool(unfinished[trajectory_index]),
                    "response_tokens": sum(response_masks[trajectory_index]),
                })
            print(
                json.dumps(
                    {
                        "pool": pool_name,
                        "prompt": index + 1,
                        "prompts": limit,
                        "successes": success_count,
                        "reward_std": reward_std,
                        "shaped_reward_std": shaped_reward_std,
                        "elapsed_seconds": time.time() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        summary = {
            "pool": pool_name,
            "seed": seed,
            "group_size": group_size,
            "data_path": str(data_path.relative_to(ROOT)),
            "data_sha256": sha256_file(data_path),
            **aggregate_probe_groups(groups),
            "wall_time_seconds": time.time() - started,
        }
        all_summaries[pool_name] = summary
        with (output_dir / f"{pool_name}_groups.jsonl").open("w", encoding="utf-8") as handle:
            for row in groups:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / f"{pool_name}_trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for row in trajectories:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
