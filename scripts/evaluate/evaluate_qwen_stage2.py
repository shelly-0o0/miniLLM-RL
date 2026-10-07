"""Evaluate the four-run Qwen3 Stage 2 matrix on one frozen split."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.gsm8k import sha256_file
from dataset.lm_dataset import AgentRLDataset
from trainer.agent_chat import render_agent_chat
from trainer.qwen3_adapter import (
    load_base_model,
    load_policy_model,
    load_stage2_tokenizer,
    load_yaml_config,
    require_section,
)
from trainer.rollout_engine import TorchRolloutEngine
from trainer.train_agent import calculate_rewards, rollout_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 Stage 2 matrix")
    parser.add_argument(
        "--config", default=str(ROOT / "configs/qwen3_4b/eval_matrix.yaml")
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override evaluation.limit. Use 0 to evaluate the entire split.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Evaluate exactly one configured arm in an isolated shard directory. "
            "Use merge_qwen_stage2_shards.py to create the canonical matrix result."
        ),
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix for a non-formal shard smoke run.",
    )
    return parser.parse_args()


def collate(rows):
    return {
        "messages": [row["messages"] for row in rows],
        "tools": [row["tools"] for row in rows],
        "required_tools": [row["required_tools"] for row in rows],
        "gt": [row["gt"] for row in rows],
    }


def mean(values):
    return sum(values) / max(len(values), 1)


def safe_path_component(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{field} contains unsafe path characters: {value!r}")
    return value


def main():
    cli = parse_args()
    config = load_yaml_config(Path(cli.config).resolve())
    model_config = require_section(config, "model")
    lora_config = require_section(config, "lora")
    evaluation = require_section(config, "evaluation")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("eval config requires a non-empty runs list")
    matrix_labels = [run["label"] for run in runs]
    if len(matrix_labels) != len(set(matrix_labels)):
        raise ValueError(f"eval config contains duplicate labels: {matrix_labels}")
    if cli.run_label is not None:
        label = safe_path_component(cli.run_label, "run label")
        runs = [run for run in runs if run["label"] == label]
        if len(runs) != 1:
            raise ValueError(
                f"run label {label!r} is not present exactly once in {matrix_labels}"
            )
    elif cli.output_tag is not None:
        raise ValueError("--output-tag requires --run-label")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 evaluation requires CUDA")

    data_path = ROOT / evaluation[f"{cli.split}_path"]
    if cli.run_label is None:
        output_dir = ROOT / evaluation["output_dir"] / cli.split
    else:
        shard_name = f"{cli.split}_shards"
        if cli.output_tag:
            shard_name += f"_{safe_path_component(cli.output_tag, 'output tag')}"
        output_dir = ROOT / evaluation["output_dir"] / shard_name / runs[0]["label"]
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"formal evaluation already exists: {summary_path}; archive it before rerunning"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(cli.config).resolve()
    (output_dir / "used_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    limit = int(evaluation.get("limit", 0)) if cli.limit is None else cli.limit
    seeds = [int(seed) for seed in evaluation.get("decode_seeds", [42])]
    device = evaluation.get("device", "cuda:0")
    all_summaries = []
    require_all_runs = bool(evaluation.get("require_all_runs", False))
    data_sha256 = sha256_file(data_path)
    manifest = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "split": cli.split,
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": data_sha256,
        "limit": limit,
        "decode_seeds": seeds,
        "sampling": {
            key: evaluation.get(key)
            for key in (
                "max_prompt_length", "max_turns", "max_new_tokens",
                "temperature", "top_k", "top_p",
            )
        },
        "require_all_runs": require_all_runs,
        "matrix_labels": matrix_labels,
        "selected_labels": [run["label"] for run in runs],
        "sharded": cli.run_label is not None,
        "runs": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for run in runs:
        label = run["label"]
        adapter_path = run.get("adapter_path")
        if adapter_path:
            adapter_path = ROOT / adapter_path
            if not adapter_path.exists():
                if require_all_runs:
                    raise FileNotFoundError(
                        f"required evaluation adapter is missing: {adapter_path}"
                    )
                print(f"Skipping {label}: missing adapter {adapter_path}")
                continue
            adapter_model_path = adapter_path / "adapter_model.safetensors"
            if not adapter_model_path.is_file():
                raise FileNotFoundError(
                    f"required adapter weights are missing: {adapter_model_path}"
                )
            adapter_sha256 = sha256_file(adapter_model_path)
            tokenizer_source = str(adapter_path)
        else:
            adapter_sha256 = None
            tokenizer_source = model_config["name"]
        tokenizer = load_stage2_tokenizer(
            tokenizer_source,
            cache_dir=model_config.get("cache_dir"),
            padding_side="left",
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            use_im_end_as_eos=bool(model_config.get("use_im_end_as_eos", True)),
        )
        if adapter_path:
            model = load_policy_model(
                model_config, lora_config, device=device,
                adapter_path=str(adapter_path), trainable=False,
            )
        else:
            model = load_base_model(model_config, device=device, for_training=False)
        model.eval()
        engine = TorchRolloutEngine(model, tokenizer, device=device)
        dataset = AgentRLDataset(data_path, tokenizer, max_length=int(evaluation.get("max_prompt_length", 1024)))
        rows = dataset.samples[:limit] if limit else dataset.samples
        # Keep AgentRLDataset's parsing/fallback semantics while selecting a
        # deterministic prefix from the frozen manifest.
        indices = range(len(rows))
        records = []
        started = time.time()
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            for index in indices:
                item = dataset[index]
                batch = collate([item])
                values = rollout_batch(
                    engine, tokenizer, batch["messages"], batch["tools"], 1,
                    max_turns=int(evaluation.get("max_turns", 3)),
                    max_new_tokens=int(evaluation.get("max_new_tokens", 384)),
                    thinking_ratio=0.0,
                    temperature=float(evaluation.get("temperature", 0.7)),
                    top_k=int(evaluation.get("top_k", 20)),
                    top_p=float(evaluation.get("top_p", 0.95)),
                    device=device,
                )
                completions, _, _, response_ids, response_masks, _, turn_outputs, unfinished, traces = values
                prompt = render_agent_chat(
                    tokenizer, item["messages"], tools=item["tools"],
                    tokenize=False, add_generation_prompt=True,
                    open_thinking=False,
                )
                reward = calculate_rewards(
                    [prompt], completions, [item["gt"]], [item["tools"]], 1,
                    [item["required_tools"]], reward_model=None, device=device,
                    turn_outputs_batch=turn_outputs,
                    unfinished_batch=unfinished,
                    require_tool_call_for_success=True,
                    reward_mode="strict",
                    return_details=True,
                )
                records.append({
                    "label": label,
                    "seed": seed,
                    "index": index,
                    "gt": item["gt"],
                    "completion": completions[0],
                    "turn_outputs": turn_outputs[0],
                    "trace": traces[0],
                    "response_tokens": sum(response_masks[0]),
                    "unfinished": bool(unfinished[0]),
                    "reward": reward.rewards[0].item(),
                    "task_success": bool(reward.task_success[0]),
                    "answer_accuracy": reward.answer_accuracy[0].item(),
                    "format_valid": reward.format_valid[0].item(),
                    "tool_call_valid": reward.tool_call_valid[0].item(),
                    "tool_execution_success": reward.tool_execution_success[0].item(),
                    "required_tool_coverage": reward.required_tool_coverage[0].item(),
                    "tool_evidence_coverage": reward.tool_evidence_coverage[0].item(),
                })
        trajectory_path = output_dir / f"{label}_trajectories.jsonl"
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = {
            "label": label,
            "split": cli.split,
            "model_name": model_config["name"],
            "model_revision": model_config.get("revision"),
            "adapter_path": (
                str(adapter_path.relative_to(ROOT)) if adapter_path else None
            ),
            "adapter_model_sha256": adapter_sha256,
            "data_path": str(data_path.relative_to(ROOT)),
            "data_sha256": data_sha256,
            "decode_seeds": seeds,
            "num_trajectories": len(records),
            "reward": mean([row["reward"] for row in records]),
            "task_accuracy": mean([row["task_success"] for row in records]),
            "answer_accuracy": mean([row["answer_accuracy"] for row in records]),
            "format_valid_rate": mean([row["format_valid"] for row in records]),
            "tool_call_valid_rate": mean([row["tool_call_valid"] for row in records]),
            "tool_execution_success_rate": mean([row["tool_execution_success"] for row in records]),
            "required_tool_coverage_rate": mean([row["required_tool_coverage"] for row in records]),
            "tool_evidence_coverage_rate": mean([row["tool_evidence_coverage"] for row in records]),
            "unfinished_rate": mean([row["unfinished"] for row in records]),
            "avg_response_tokens": mean([row["response_tokens"] for row in records]),
            "wall_time_seconds": time.time() - started,
        }
        all_summaries.append(summary)
        manifest["runs"].append(
            {
                "label": label,
                "adapter_path": summary["adapter_path"],
                "adapter_model_sha256": adapter_sha256,
                "trajectory_path": str(trajectory_path.relative_to(ROOT)),
                "num_trajectories": len(records),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del engine, model
        gc.collect()
        torch.cuda.empty_cache()

    if require_all_runs and len(all_summaries) != len(runs):
        raise RuntimeError(
            f"evaluation produced {len(all_summaries)} of {len(runs)} required runs"
        )
    summary_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["status"] = "COMPLETE"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_path"] = str(summary_path.relative_to(ROOT))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
