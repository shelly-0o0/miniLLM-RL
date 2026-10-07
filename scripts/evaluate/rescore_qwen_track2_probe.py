"""Re-score saved Stage 2 rollouts after a deterministic reward-code change."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.probe_qwen_track2 import aggregate_probe_groups
from trainer.train_agent import calculate_rewards


def read_jsonl(path: Path) -> list[dict]:
    # Iterate physical ``\n``-delimited records. ``str.splitlines`` also splits
    # Unicode line separators that can legitimately occur inside a JSON string.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--pool", default="a")
    parser.add_argument(
        "--data-path",
        default="data/processed/gsm8k_track2/a_probe_rl.jsonl",
    )
    parser.add_argument("--output-tag", default="reward_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = (ROOT / args.probe_dir).resolve()
    data_path = (ROOT / args.data_path).resolve()
    source_by_id = {row["id"]: row for row in read_jsonl(data_path)}
    saved = read_jsonl(probe_dir / f"{args.pool}_trajectories.jsonl")
    by_index: dict[int, list[dict]] = defaultdict(list)
    for row in saved:
        by_index[int(row["index"])].append(row)

    rescored_groups, rescored_trajectories = [], []
    for index in sorted(by_index):
        trajectories = sorted(
            by_index[index], key=lambda row: int(row["trajectory_index"])
        )
        sample = source_by_id[trajectories[0]["id"]]
        group_size = len(trajectories)
        kwargs = dict(
            prompts=[""],
            completions=[row["completion"] for row in trajectories],
            gt_batch=[sample["gt"]],
            tools_batch=[sample["tools"]],
            num_gen=group_size,
            required_tools_batch=[sample.get("required_tools", [])],
            device="cpu",
            turn_outputs_batch=[row["turn_outputs"] for row in trajectories],
            unfinished_batch=[bool(row["unfinished"]) for row in trajectories],
            require_tool_call_for_success=True,
            return_details=True,
        )
        strict = calculate_rewards(reward_mode="strict", **kwargs)
        shaped = calculate_rewards(reward_mode="shaped", **kwargs)
        success_count = int(strict.task_success.sum().item())
        strict_std = strict.rewards.float().std(unbiased=False).item()
        shaped_std = shaped.rewards.float().std(unbiased=False).item()
        rescored_groups.append({
            "pool": args.pool,
            "index": index,
            "id": sample["id"],
            "num_trajectories": group_size,
            "success_count": success_count,
            "pass_at_1": float(strict.task_success[0].item()),
            "pass_at_k": float(success_count > 0),
            "effective_group": float(0 < success_count < group_size),
            "zero_variance_group": float(strict_std < 1e-6),
            "reward_std": strict_std,
            "shaped_reward_mean": shaped.rewards.mean().item(),
            "shaped_reward_std": shaped_std,
            "shaped_nonzero_variance_group": float(shaped_std >= 1e-6),
            "answer_accuracy": strict.answer_accuracy.mean().item(),
            "format_valid_rate": strict.format_valid.mean().item(),
            "tool_call_valid_rate": strict.tool_call_valid.mean().item(),
            "tool_execution_success_rate": strict.tool_execution_success.mean().item(),
            "required_tool_coverage_rate": strict.required_tool_coverage.mean().item(),
            "tool_evidence_coverage_rate": strict.tool_evidence_coverage.mean().item(),
            "protocol_progress": shaped.protocol_progress.mean().item(),
            "unfinished_rate": sum(bool(row["unfinished"]) for row in trajectories) / group_size,
            "avg_response_tokens": sum(int(row["response_tokens"]) for row in trajectories) / group_size,
        })
        for position, row in enumerate(trajectories):
            rescored_trajectories.append({
                **row,
                "reward": strict.rewards[position].item(),
                "shaped_reward": shaped.rewards[position].item(),
                "task_success": bool(strict.task_success[position].item()),
                "format_valid": strict.format_valid[position].item(),
                "tool_call_valid": strict.tool_call_valid[position].item(),
                "tool_execution_success": strict.tool_execution_success[position].item(),
                "required_tool_coverage": strict.required_tool_coverage[position].item(),
                "tool_evidence_coverage": strict.tool_evidence_coverage[position].item(),
                "protocol_progress": shaped.protocol_progress[position].item(),
            })

    summary = {
        "pool": args.pool,
        "reward_version": args.output_tag,
        **aggregate_probe_groups(rescored_groups),
    }
    write_jsonl(probe_dir / f"{args.pool}_groups_{args.output_tag}.jsonl", rescored_groups)
    write_jsonl(
        probe_dir / f"{args.pool}_trajectories_{args.output_tag}.jsonl",
        rescored_trajectories,
    )
    output = probe_dir / f"summary_{args.output_tag}.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
