"""Create a deterministic, question-grouped Agent RL train/eval split."""

import argparse
import hashlib
import json
import os
import random
import unicodedata


def question_key(sample):
    user_turns = [
        str(message.get("content", ""))
        for message in sample.get("conversations", [])
        if message.get("role") == "user"
    ]
    normalized = unicodedata.normalize("NFKC", "\n".join(user_turns)).casefold()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Split Agent JSONL without exact-question leakage")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--eval_output", required=True)
    parser.add_argument("--manifest", default="./out/run_meta/agent_split_manifest.json")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        parser.error("--eval_ratio must be between 0 and 1")

    rows = []
    with open(args.input, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not row.get("conversations") or "gt" not in row:
                    raise ValueError(f"invalid Agent sample at line {line_number}")
                rows.append(row)
    if len(rows) < 2:
        raise ValueError("at least two samples are required")

    groups = {}
    for row in rows:
        groups.setdefault(question_key(row), []).append(row)
    if len(groups) < 2:
        raise ValueError("at least two unique user-question groups are required")
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    eval_group_count = min(len(keys) - 1, max(1, round(len(keys) * args.eval_ratio)))
    eval_keys = set(keys[:eval_group_count])
    train_rows = [row for key in keys if key not in eval_keys for row in groups[key]]
    eval_rows = [row for key in keys if key in eval_keys for row in groups[key]]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    manifest = {
        "input": os.path.abspath(args.input),
        "input_sha256": file_sha256(args.input),
        "seed": args.seed,
        "eval_ratio_requested": args.eval_ratio,
        "total_samples": len(rows),
        "unique_question_groups": len(keys),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "train_question_hashes": sorted(set(keys) - eval_keys),
        "eval_question_hashes": sorted(eval_keys),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in manifest.items() if not key.endswith("hashes")}, indent=2))


if __name__ == "__main__":
    main()
