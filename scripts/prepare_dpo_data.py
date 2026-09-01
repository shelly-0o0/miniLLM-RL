"""Create a deterministic prompt-grouped train/eval split for MiniMind DPO."""

import argparse
import hashlib
import json
import os
import random
import unicodedata


def canonical_prompt(row):
    chosen = row.get("chosen", [])
    prefix = chosen[:-1] if chosen and chosen[-1].get("role") == "assistant" else chosen
    text = json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split DPO pairs without exact-prompt leakage")
    parser.add_argument("--input", default="dataset/dpo.jsonl")
    parser.add_argument("--train_output", default="dataset/dpo_train.jsonl")
    parser.add_argument("--eval_output", default="dataset/dpo_eval.jsonl")
    parser.add_argument("--manifest", default="out/run_meta/dpo_split_manifest.json")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.eval_ratio < 1:
        parser.error("eval_ratio must be in (0, 1)")

    groups = {}
    with open(args.input, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("chosen") or not row.get("rejected"):
                raise ValueError(f"invalid preference pair at line {line_number}")
            groups.setdefault(sha256_text(canonical_prompt(row)), []).append(row)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    eval_groups = max(1, min(len(keys) - 1, round(len(keys) * args.eval_ratio)))
    eval_keys = set(keys[:eval_groups])
    train = [row for key in keys if key not in eval_keys for row in groups[key]]
    evaluate = [row for key in keys if key in eval_keys for row in groups[key]]
    write_jsonl(args.train_output, train)
    write_jsonl(args.eval_output, evaluate)
    manifest = {
        "input": os.path.abspath(args.input),
        "input_sha256": file_sha256(args.input),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "unique_prompt_groups": len(keys),
        "train_samples": len(train),
        "eval_samples": len(evaluate),
        "prompt_overlap": 0,
        "train_sha256": file_sha256(args.train_output),
        "eval_sha256": file_sha256(args.eval_output),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
