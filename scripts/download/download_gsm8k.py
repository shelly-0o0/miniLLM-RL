"""Download openai/gsm8k to auditable local JSONL files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.gsm8k import normalize_rows, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/gsm8k")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the 'datasets' package from requirements.txt first") from exc

    dataset = load_dataset("openai/gsm8k", "main")
    output = Path(args.output_dir)
    for split in ("train", "test"):
        path = output / f"official_{split}.jsonl"
        count = write_jsonl(path, normalize_rows(dataset[split], split))
        print(f"wrote {count} rows to {path}")


if __name__ == "__main__":
    main()
