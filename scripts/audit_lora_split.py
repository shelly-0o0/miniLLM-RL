"""Verify and fingerprint the persisted LoRA train/holdout split."""

import argparse
import hashlib
import json
import os


def canonical_rows(path):
    digest = hashlib.sha256()
    rows = []
    with open(path, "rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return rows, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset/lora_medical.jsonl")
    parser.add_argument("--train", default="dataset/lora_medical_train.jsonl")
    parser.add_argument("--eval", default="dataset/lora_medical_eval.jsonl")
    parser.add_argument("--output", default="out/run_meta/lora_medical_split_manifest.json")
    args = parser.parse_args()
    source, source_sha = canonical_rows(args.source)
    train, train_sha = canonical_rows(args.train)
    evaluation, eval_sha = canonical_rows(args.eval)
    source_set, train_set, eval_set = set(source), set(train), set(evaluation)
    report = {
        "source": {"path": os.path.abspath(args.source), "rows": len(source), "unique": len(source_set), "sha256": source_sha},
        "train": {"path": os.path.abspath(args.train), "rows": len(train), "unique": len(train_set), "sha256": train_sha},
        "eval": {"path": os.path.abspath(args.eval), "rows": len(evaluation), "unique": len(eval_set), "sha256": eval_sha},
        "exact_train_eval_overlap": len(train_set & eval_set),
        "union_equals_source": train_set | eval_set == source_set,
        "split_files_are_source_order_subsequences": (
            train == [row for row in source if row in train_set]
            and evaluation == [row for row in source if row in eval_set]
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["exact_train_eval_overlap"] or not report["union_equals_source"]:
        raise SystemExit("LoRA split audit failed")


if __name__ == "__main__":
    main()
