"""Replay every generated Tool-Use RLVR oracle against the training sandbox."""

import argparse
import hashlib
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.train_agent import CHECK_ARGS, execute_tool, validate_gt_in_text


def rows(path):
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def question_hash(row):
    question = next(message["content"] for message in row["conversations"] if message["role"] == "user")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def audit(path):
    failures = []
    hashes = set()
    count = 0
    for index, row in rows(path):
        count += 1
        hashes.add(question_hash(row))
        tools = json.loads(row["conversations"][0]["tools"])
        available = {tool["function"]["name"] for tool in tools}
        required = set(row.get("required_tools") or [])
        calls = row.get("oracle_calls") or []
        expected = row.get("oracle_observations") or []
        results = []
        reasons = []
        if not calls or len(calls) != len(expected):
            reasons.append("missing_or_unpaired_oracle")
        for call_index, call in enumerate(calls):
            name, arguments = call.get("name"), call.get("arguments") or {}
            if name not in available:
                reasons.append(f"call_{call_index}_not_available")
                continue
            validator = CHECK_ARGS.get(name)
            if validator is None or not validator(arguments):
                reasons.append(f"call_{call_index}_invalid_arguments")
                continue
            result = execute_tool(name, arguments)
            results.append(result)
            if call_index >= len(expected) or result != expected[call_index]:
                reasons.append(f"call_{call_index}_observation_mismatch")
        if not required.issubset({call.get("name") for call in calls}):
            reasons.append("required_tool_not_in_oracle")
        evidence = validate_gt_in_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results),
            row.get("gt") or [],
            final_answer_only=False,
        )
        if len(evidence) != len(row.get("gt") or []):
            reasons.append("execution_evidence_does_not_cover_gt")
        if reasons:
            failures.append({"index": index, "reasons": reasons})
    return {"path": os.path.abspath(path), "rows": count, "question_hashes": hashes, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/agent_rl_tool_verified_train.jsonl")
    parser.add_argument("--eval", default="dataset/agent_rl_tool_verified_eval.jsonl")
    parser.add_argument("--output", default="out/run_meta/tool_rlvr_execution_audit.json")
    args = parser.parse_args()
    train = audit(args.train)
    evaluation = audit(args.eval)
    overlap = train.pop("question_hashes") & evaluation.pop("question_hashes")
    report = {
        "train": train,
        "eval": evaluation,
        "question_overlap": len(overlap),
        "total_failures": len(train["failures"]) + len(evaluation["failures"]),
        "checks": [
            "tool availability", "argument legality", "deterministic replay",
            "oracle observation equality", "required-tool coverage",
            "execution-result ground-truth coverage", "train/eval question isolation",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["total_failures"] or report["question_overlap"]:
        raise SystemExit("Tool RLVR dataset audit failed")


if __name__ == "__main__":
    main()
