"""Convert prepared GSM8K splits into Agent SFT and RLVR trajectories.

GSM8K reasoning contains calculator annotations of the form
``<<expression=result>>``.  We replay every annotation with the shared safe
calculator before it may become an oracle tool trajectory.  Only train rows
produce supervised trajectories; validation and test contain prompts plus
verifier-only labels.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.gsm8k import read_jsonl, write_jsonl
from trainer.math_env import canonical_number, safe_calculate


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "Evaluate a numeric arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
SYSTEM_PROMPT = (
    "Solve the math word problem. Use calculate_math for arithmetic, then give "
    "the final numeric answer explicitly as 'Final answer: <number>'."
)
ANNOTATION = re.compile(r"<<(.+?)=([^<>]+)>>")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_annotations(answer):
    calls, observations = [], []
    for expression, expected in ANNOTATION.findall(answer):
        expression, expected = expression.strip(), expected.strip().replace(",", "")
        result = safe_calculate(expression)["result"]
        actual_number, expected_number = canonical_number(result), canonical_number(expected)
        tolerance = max(abs(expected_number) * canonical_number("1e-12"), canonical_number("1e-12"))
        if abs(actual_number - expected_number) > tolerance:
            raise ValueError(f"oracle mismatch: {expression} -> {result}, stored {expected}")
        calls.append({"name": "calculate_math", "arguments": {"expression": expression}})
        observations.append({"result": result})
    return calls, observations


def prompt_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT, "tools": json.dumps([CALCULATOR_TOOL])},
        {"role": "user", "content": question},
    ]


def make_rl_row(row):
    calls, observations = replay_annotations(row["answer"])
    conversations = prompt_messages(row["question"])
    # AgentRLDataset intentionally drops the last item when constructing the
    # rollout prompt.  This sentinel carries no gold label into model input.
    conversations.append({"role": "assistant", "content": ""})
    return {
        "id": row["id"],
        "conversations": conversations,
        "tools": [CALCULATOR_TOOL],
        "required_tools": ["calculate_math"],
        "gt": [row["gold_answer"]],
        "oracle_calls": calls,
        "oracle_observations": observations,
        "source_split": row["split"],
    }


def make_sft_row(row):
    calls, observations = replay_annotations(row["answer"])
    if not calls:
        return None
    conversations = prompt_messages(row["question"])
    conversations.append({
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": json.dumps(calls, ensure_ascii=False),
    })
    conversations.extend(
        {"role": "tool", "content": json.dumps(observation)}
        for observation in observations
    )
    conversations.append({
        "role": "assistant",
        "content": f"Final answer: {row['gold_answer']}",
        "reasoning_content": "",
    })
    return {"id": row["id"], "conversations": conversations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/gsm8k")
    parser.add_argument("--output-dir", default="data/processed/gsm8k_agent")
    parser.add_argument("--manifest", default="dataset/manifests/gsm8k_agent.json")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    paths, counts = {}, {}
    all_ids = {}
    for split in ("train", "validation", "test"):
        rows = list(read_jsonl(input_dir / f"{split}.jsonl"))
        rl_rows = []
        for row in rows:
            if row["id"] in all_ids:
                raise ValueError(f"id leakage: {row['id']} in {all_ids[row['id']]} and {split}")
            all_ids[row["id"]] = split
            rl_rows.append(make_rl_row(row))
        path = output_dir / f"{split}_rl.jsonl"
        write_jsonl(path, rl_rows)
        paths[f"{split}_rl"] = path
        counts[f"{split}_rl"] = len(rl_rows)

        if split == "train":
            sft_rows = [converted for row in rows if (converted := make_sft_row(row))]
            sft_path = output_dir / "train_sft.jsonl"
            write_jsonl(sft_path, sft_rows)
            paths["train_sft"] = sft_path
            counts["train_sft"] = len(sft_rows)

    manifest = {
        "source_manifest": "dataset/manifests/gsm8k.json",
        "policy": "SFT labels from train only; validation/test labels are verifier-only",
        "calculator": "trainer.math_env.safe_calculate",
        "files": {
            name: {"path": str(path), "rows": counts[name], "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
