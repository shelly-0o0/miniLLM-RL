"""Measure per-question learnability before a short DAPO diagnostic run.

The first decode pass is used only for selecting mixed-success questions.  The
selected questions are decoded again with an independent seed before their
baseline is reported, so selection noise cannot become the claimed initial
accuracy.
"""

import argparse
import collections
import hashlib
import json
import os
import sys
from contextlib import nullcontext

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from trainer.rollout_engine import TorchRolloutEngine
from trainer.train_agent import calculate_rewards, parse_tool_calls, rollout_batch
from trainer.trainer_utils import init_model, setup_seed


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_key(row):
    messages = row.get("conversations", [])
    question = [
        str(message.get("content", ""))
        for message in messages if message.get("role") == "user"
    ]
    return hashlib.sha256("\n".join(question).strip().encode("utf-8")).hexdigest()


def row_key(row):
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def row_to_item(row):
    messages = []
    tools = None
    for message in row["conversations"][:-1]:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = (
                json.loads(message["tools"])
                if isinstance(message["tools"], str) else message["tools"]
            )
        messages.append(message)
    required_tools = list(row.get("required_tools") or [])
    available_names = {
        tool.get("function", {}).get("name") for tool in (tools or [])
    }
    if not required_tools and row.get("gt") and "calculate_math" in available_names:
        required_tools = ["calculate_math"]
    return {
        "messages": messages,
        "tools": tools,
        "required_tools": required_tools,
        "gt": row["gt"],
    }


def failure_reason(success, unfinished, format_valid, tool_call_valid,
                   tool_execution_success, required_coverage,
                   evidence_coverage, answer_accuracy, has_tools):
    if success:
        return "success"
    if unfinished:
        return "unfinished"
    if has_tools and tool_call_valid == 0:
        return "ignored_feedback_no_tool_call"
    if format_valid == 0:
        return "format_error"
    if tool_call_valid < 1 or tool_execution_success < 1:
        return "parameter_or_tool_execution_error"
    if required_coverage < 1 or evidence_coverage < 1:
        return "ignored_feedback_missing_required_tool_or_evidence"
    if answer_accuracy == 0:
        return "wrong_final_answer"
    return "other_failure"


def balanced_rows(rows, limit, seed):
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row.get("rlvr_category", "uncategorized")].append(row)
    for category, values in buckets.items():
        values.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row_key(row)}".encode("utf-8")
        ).hexdigest())
    selected = []
    categories = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if buckets[category] and len(selected) < limit:
                selected.append(buckets[category].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


@torch.no_grad()
def inspect_rows(rows, model, tokenizer, args, seed):
    """Return one aggregate record and all raw trajectory records per row."""

    setup_seed(seed)
    engine = TorchRolloutEngine(model, tokenizer, args.device, autocast_ctx=nullcontext())
    aggregate = {}
    trajectories = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start:batch_start + args.batch_size]
        items = [row_to_item(row) for row in batch_rows]
        messages = [item["messages"] for item in items]
        tools = [item["tools"] for item in items]
        gt = [item["gt"] for item in items]
        required = [item["required_tools"] for item in items]
        prompts = [
            tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True,
                tools=item["tools"],
            )
            for item in items
        ]
        rollout_values = rollout_batch(
            engine, tokenizer, messages, tools, args.num_generations,
            max_turns=args.max_turns, max_new_tokens=args.max_gen_len,
            thinking_ratio=args.thinking_ratio,
            temperature=args.rollout_temperature,
            top_k=args.rollout_top_k, top_p=args.rollout_top_p,
            device=args.device,
        )
        (completions, _, _, response_ids, response_masks, _, turn_outputs,
         unfinished, traces) = rollout_values
        reward = calculate_rewards(
            prompts, completions, gt, tools, args.num_generations, required,
            device=args.device, turn_outputs_batch=turn_outputs,
            unfinished_batch=unfinished, require_tool_call_for_success=True,
            reward_mode="strict", return_details=True,
        )
        lengths = [sum(mask) for mask in response_masks]
        for local_index, row in enumerate(batch_rows):
            key = row_key(row)
            aggregate.setdefault(key, {
                "task_id": row.get("task_id", key[:12]),
                "row_key": key,
                "category": row.get("rlvr_category", "uncategorized"),
                "question_key": question_key(row),
                "row": row,
                "successes": 0,
                "num_generations": 0,
                "failure_reasons": collections.Counter(),
                "trajectory_hashes": set(),
                "generated_tokens": 0,
                "tool_calls": 0,
            })
            record = aggregate[key]
            for generation_index in range(args.num_generations):
                index = local_index * args.num_generations + generation_index
                success = bool(reward.task_success[index])
                reason = failure_reason(
                    success, bool(unfinished[index]),
                    float(reward.format_valid[index]),
                    float(reward.tool_call_valid[index]),
                    float(reward.tool_execution_success[index]),
                    float(reward.required_tool_coverage[index]),
                    float(reward.tool_evidence_coverage[index]),
                    float(reward.answer_accuracy[index]), bool(tools[local_index]),
                )
                trace = traces[index]
                record["successes"] += int(success)
                record["num_generations"] += 1
                record["failure_reasons"][reason] += 1
                record["trajectory_hashes"].add(hashlib.sha256(
                    json.dumps(trace, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest())
                record["generated_tokens"] += int(lengths[index])
                trajectory_tool_calls = sum(
                    len(parse_tool_calls(turn)) for turn in turn_outputs[index]
                )
                record["tool_calls"] += trajectory_tool_calls
                trajectories.append({
                    "task_id": record["task_id"],
                    "category": record["category"],
                    "question_key": record["question_key"],
                    "seed": seed,
                    "generation_index": generation_index,
                    "success": success,
                    "failure_reason": reason,
                    "generated_tokens": int(lengths[index]),
                    "tool_calls": trajectory_tool_calls,
                    "trace": trace,
                    "reward": float(reward.rewards[index]),
                })
    result = []
    for record in aggregate.values():
        total = record["num_generations"]
        record["success_rate"] = record["successes"] / max(total, 1)
        record["major_failure_reason"] = (
            record["failure_reasons"].most_common(1)[0][0]
            if record["failure_reasons"] else "none"
        )
        record["distinct_trajectories"] = len(record["trajectory_hashes"])
        record["average_generated_tokens"] = record["generated_tokens"] / max(total, 1)
        record["average_tool_calls"] = record["tool_calls"] / max(total, 1)
        record["mixed_success"] = 0 < record["successes"] < total
        record["failure_reasons"] = dict(record["failure_reasons"])
        record["trajectory_hashes"] = sorted(record["trajectory_hashes"])
        result.append(record)
    return result, trajectories


def main():
    parser = argparse.ArgumentParser(description="Inspect Agent-RL question learnability")
    parser.add_argument("--data_path", default="dataset/agent_rl_tool_challenge_train.jsonl")
    parser.add_argument("--sft_data_path", default="dataset/agent_sft_tool_challenge_curriculum_96.jsonl")
    parser.add_argument("--weight", default="agent_sft_tool_curriculum96")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--output_dir", default="out/learnability")
    parser.add_argument("--num_questions", type=int, default=96)
    parser.add_argument("--diagnostic_questions", type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--selection_seed", type=int, default=20260901)
    parser.add_argument("--independent_seed", type=int, default=20260902)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--thinking_ratio", type=float, default=0.0)
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_top_k", type=int, default=0)
    parser.add_argument("--rollout_top_p", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_generations < 2 or args.num_questions < 1:
        parser.error("num_generations must be >= 2 and num_questions must be >= 1")
    if args.rollout_temperature <= 0 or args.rollout_top_k < 0 or not 0 < args.rollout_top_p <= 1:
        parser.error("invalid rollout sampling configuration")

    rows = read_jsonl(args.data_path)
    sft_rows = read_jsonl(args.sft_data_path) if os.path.exists(args.sft_data_path) else []
    excluded = {question_key(row) for row in sft_rows}
    pool = [row for row in rows if question_key(row) not in excluded]
    ordered = balanced_rows(pool, len(pool), args.selection_seed)

    config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_position_embeddings,
    )
    model, tokenizer = init_model(
        config, args.weight, tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir, device=args.device,
    )
    model.eval()

    inspected_rows = ordered[:args.num_questions]
    selected_records, selection_trajectories = inspect_rows(
        inspected_rows, model, tokenizer, args, args.selection_seed
    )
    mixed = [record for record in selected_records if record["mixed_success"]]
    if len(mixed) < args.diagnostic_questions:
        remaining = ordered[args.num_questions:]
        extra_records, extra_trajectories = inspect_rows(
            remaining, model, tokenizer, args, args.selection_seed
        )
        selection_trajectories.extend(extra_trajectories)
        selected_records.extend(extra_records)
        mixed.extend(record for record in extra_records if record["mixed_success"])
    mixed = mixed[:args.diagnostic_questions]
    selected_rows = [record["row"] for record in mixed]

    # Independent decode: this is the only baseline used for the selected set.
    independent_records, independent_trajectories = inspect_rows(
        selected_rows, model, tokenizer, args, args.independent_seed
    ) if selected_rows else ([], [])
    independent_by_key = {
        record["row_key"]: record for record in independent_records
    }
    for record in selected_records:
        independent = independent_by_key.get(record["row_key"])
        if independent:
            record["independent_successes"] = independent["successes"]
            record["independent_success_rate"] = independent["success_rate"]

    categories = collections.Counter(record["category"] for record in selected_records)
    summary = {
        "data_path": os.path.abspath(args.data_path),
        "sft_data_path": os.path.abspath(args.sft_data_path),
        "excluded_sft_questions": len(excluded),
        "candidate_pool": len(pool),
        "initial_inspection_questions": len(inspected_rows),
        "inspected_questions_total": len(selected_records),
        "selected_diagnostic_questions": len(selected_rows),
        "selected_category_counts": dict(sorted(collections.Counter(
            row.get("rlvr_category", "uncategorized") for row in selected_rows
        ).items())),
        "all_correct_questions": sum(record["successes"] == record["num_generations"] for record in selected_records),
        "all_wrong_questions": sum(record["successes"] == 0 for record in selected_records),
        "mixed_success_questions": sum(record["mixed_success"] for record in selected_records),
        "selection_seed": args.selection_seed,
        "independent_seed": args.independent_seed,
        "sampling": {
            "temperature": args.rollout_temperature,
            "top_k": args.rollout_top_k,
            "top_p": args.rollout_top_p,
            "num_generations": args.num_generations,
        },
        "note": "selection decode is not the baseline; independent_seed records are the selected-set baseline",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "question_report.jsonl"), "w", encoding="utf-8") as handle:
        for record in sorted(selected_records, key=lambda item: item["task_id"]):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with open(os.path.join(args.output_dir, "selection_trajectories.jsonl"), "w", encoding="utf-8") as handle:
        for record in selection_trajectories:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with open(os.path.join(args.output_dir, "independent_trajectories.jsonl"), "w", encoding="utf-8") as handle:
        for record in independent_trajectories:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with open(os.path.join(args.output_dir, "diagnostic_train.jsonl"), "w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
