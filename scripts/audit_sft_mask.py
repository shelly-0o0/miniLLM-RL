"""Audit MiniMind assistant-only SFT labels and emit inspectable evidence."""

import argparse
import json
import os
import random
import sys

from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import SFTDataset


def expected_assistant_mask(input_ids, bos_ids, eos_ids):
    expected = [False] * len(input_ids)
    cursor = 0
    while cursor < len(input_ids):
        if input_ids[cursor:cursor + len(bos_ids)] == bos_ids:
            start = cursor + len(bos_ids)
            end = start
            while end < len(input_ids) and input_ids[end:end + len(eos_ids)] != eos_ids:
                end += 1
            stop = min(end + len(eos_ids), len(input_ids))
            for position in range(start, stop):
                expected[position] = True
            cursor = stop
        else:
            cursor += 1
    return expected


def supervised_spans(tokenizer, input_ids, labels):
    spans, current = [], []
    for token_id, label in zip(input_ids, labels):
        if label != -100:
            current.append(token_id)
        elif current:
            spans.append(tokenizer.decode(current, skip_special_tokens=False))
            current = []
    if current:
        spans.append(tokenizer.decode(current, skip_special_tokens=False))
    return spans


def main():
    parser = argparse.ArgumentParser(description="Audit MiniMind SFT assistant-only loss masks")
    parser.add_argument("--data_path", default="./dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_path", default="./model")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./out/run_meta/sft_mask_audit.json")
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sample_count = min(args.samples, len(dataset))
    if sample_count == 0:
        raise ValueError("SFT dataset is empty")
    mismatches = []
    zero_supervision = []
    supervised_tokens = 0
    nonpad_tokens = 0
    examples = []
    for index in range(sample_count):
        input_tensor, label_tensor = dataset[index]
        input_ids = input_tensor.tolist()
        labels = label_tensor.tolist()
        expected = expected_assistant_mask(input_ids, dataset.bos_id, dataset.eos_id)
        actual = [label != -100 for label in labels]
        bad_positions = [position for position, (left, right) in enumerate(zip(actual, expected)) if left != right]
        bad_label_values = [
            position for position, (token_id, label) in enumerate(zip(input_ids, labels))
            if label != -100 and label != token_id
        ]
        bad_pad = [
            position for position, (token_id, label) in enumerate(zip(input_ids, labels))
            if token_id == tokenizer.pad_token_id and label != -100
        ]
        if bad_positions or bad_label_values or bad_pad:
            mismatches.append({
                "index": index,
                "mask_positions": bad_positions[:20],
                "label_positions": bad_label_values[:20],
                "pad_positions": bad_pad[:20],
            })
        count = sum(actual)
        supervised_tokens += count
        nonpad_tokens += sum(token_id != tokenizer.pad_token_id for token_id in input_ids)
        if count == 0:
            zero_supervision.append(index)
        if len(examples) < args.examples:
            examples.append({
                "index": index,
                "supervised_token_count": count,
                "supervised_spans": supervised_spans(tokenizer, input_ids, labels),
            })

    summary = {
        "data_path": os.path.abspath(args.data_path),
        "seed": args.seed,
        "samples_audited": sample_count,
        "supervised_tokens": supervised_tokens,
        "nonpad_tokens": nonpad_tokens,
        "assistant_token_ratio": supervised_tokens / max(nonpad_tokens, 1),
        "mismatch_count": len(mismatches),
        "zero_supervision_count": len(zero_supervision),
        "mismatches": mismatches,
        "zero_supervision_indices": zero_supervision,
        "examples": examples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"examples", "mismatches"}}, ensure_ascii=False, indent=2))
    if mismatches or zero_supervision:
        raise SystemExit("SFT mask audit failed; inspect the JSON evidence")


if __name__ == "__main__":
    main()
