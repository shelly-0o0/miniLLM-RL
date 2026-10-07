"""Create fixed GSM8K train/validation/test files and a hash manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.gsm8k import assert_disjoint, build_manifest, read_jsonl, split_train_validation, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/gsm8k")
    parser.add_argument("--output-dir", default="data/processed/gsm8k")
    parser.add_argument("--manifest", default="dataset/manifests/gsm8k.json")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw, output = Path(args.raw_dir), Path(args.output_dir)
    official_train = list(read_jsonl(raw / "official_train.jsonl"))
    official_test = [{**row, "split": "test"} for row in read_jsonl(raw / "official_test.jsonl")]
    train, validation = split_train_validation(official_train, args.validation_size, args.seed)
    splits = {"train": train, "validation": validation, "test": official_test}
    assert_disjoint(splits)
    paths = {name: output / f"{name}.jsonl" for name in splits}
    for name, rows in splits.items():
        write_jsonl(paths[name], rows)
    manifest = build_manifest(paths, args.seed)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
