"""Build deterministic, deduplicated train/eval subsets for KD experiments."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def canonicalize(line):
    obj = json.loads(line)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def minhash_sample(path, size, salt):
    # Keep the `size` smallest salted SHA-256 values without loading the whole
    # source dataset.  Exact duplicates have the same canonical representation
    # and therefore cannot cross experiment splits.
    heap = []
    selected = set()
    source_rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_rows += 1
            canonical = canonicalize(line)
            if canonical in selected:
                continue
            score = int.from_bytes(
                hashlib.sha256((salt + canonical).encode("utf-8")).digest()[:8],
                "big",
            )
            item = (-score, canonical)
            if len(heap) < size:
                heapq.heappush(heap, item)
                selected.add(canonical)
            elif score < -heap[0][0]:
                _, removed = heapq.heapreplace(heap, item)
                selected.remove(removed)
                selected.add(canonical)
    rows = sorted(((-negative_score, canonical) for negative_score, canonical in heap))
    return source_rows, rows


def write_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for _, canonical in rows:
            line = canonical + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed KD train and independent eval subsets")
    parser.add_argument("--train_source", default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--eval_source", default="dataset/rlaif.jsonl")
    parser.add_argument("--train_output", default="dataset/kd_sft_train_50000.jsonl")
    parser.add_argument("--eval_output", default="dataset/kd_rlaif_eval_2000.jsonl")
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--eval_size", type=int, default=2000)
    parser.add_argument("--salt", default="minimind-kd-v1|")
    parser.add_argument("--manifest", default="out/run_meta/kd_dataset_manifest.json")
    args = parser.parse_args()

    train_source_rows, train_rows = minhash_sample(
        args.train_source, args.train_size, args.salt + "train|"
    )
    eval_source_rows, eval_rows = minhash_sample(
        args.eval_source, args.eval_size, args.salt + "eval|"
    )
    train_canonical = {canonical for _, canonical in train_rows}
    eval_canonical = {canonical for _, canonical in eval_rows}
    canonical_overlap = train_canonical & eval_canonical
    if canonical_overlap:
        raise RuntimeError(
            f"KD train/eval exact canonical overlap is {len(canonical_overlap)}; "
            "choose an independent eval source or remove duplicates before training"
        )
    train_sha = write_rows(args.train_output, train_rows)
    eval_sha = write_rows(args.eval_output, eval_rows)
    manifest = {
        "schema_version": 1,
        "method": "deduplicated salted SHA-256 min-hash sample",
        "salt": args.salt,
        "train": {
            "source": args.train_source,
            "source_rows": train_source_rows,
            "output": args.train_output,
            "rows": len(train_rows),
            "sha256": train_sha,
        },
        "eval": {
            "source": args.eval_source,
            "source_rows": eval_source_rows,
            "output": args.eval_output,
            "rows": len(eval_rows),
            "sha256": eval_sha,
        },
        "exact_canonical_overlap": len(canonical_overlap),
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
