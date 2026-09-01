"""Evaluate a MiniMind LoRA adapter on an assistant-only held-out set.

The reported loss is weighted by the number of non-ignored next-token labels,
so batches with different assistant response lengths contribute correctly.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(args, adapter_path=None):
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(args.base_weight_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if adapter_path:
        apply_lora(model, rank=args.lora_rank)
        load_lora(model, adapter_path)
    return model.eval().to(device=args.device, dtype=args.dtype)


@torch.inference_mode()
def evaluate(model, loader, args):
    total_nll = 0.0
    total_tokens = 0
    total_samples = 0
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    for input_ids, labels in loader:
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        valid_tokens = labels[..., 1:].ne(-100).sum()
        if valid_tokens.item() == 0:
            continue
        result = model(input_ids, labels=labels)
        token_count = int(valid_tokens.item())
        total_nll += float(result.loss.float().item()) * token_count
        total_tokens += token_count
        total_samples += input_ids.size(0)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
        peak_memory = torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20
    else:
        peak_memory = 0.0
    elapsed = time.perf_counter() - started
    mean_loss = total_nll / max(total_tokens, 1)
    return {
        "samples": total_samples,
        "assistant_tokens": total_tokens,
        "token_weighted_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "elapsed_seconds": elapsed,
        "assistant_tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "peak_memory_mib": peak_memory,
    }


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_dataset(dataset, seed):
    """Freeze stochastic SFT preprocessing before comparing two models."""
    random.seed(seed)
    torch.manual_seed(seed)
    input_ids, labels = zip(*(dataset[index] for index in range(len(dataset))))
    return TensorDataset(torch.stack(input_ids), torch.stack(labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Base versus LoRA on held-out SFT data")
    parser.add_argument("--eval_data", default="dataset/lora_medical_eval.jsonl")
    parser.add_argument("--base_weight_path", default="out/full_sft_768.pth")
    parser.add_argument("--adapter_path", default="out/lora_medical_holdout_768.pth")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--output", default="out/eval/lora_medical_holdout_metrics.json")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    args.dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    stochastic_dataset = SFTDataset(args.eval_data, tokenizer, max_length=args.max_seq_len)
    dataset = materialize_dataset(stochastic_dataset, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    base_model = load_model(args)
    base_metrics = evaluate(base_model, loader, args)
    release(base_model)

    lora_model = load_model(args, args.adapter_path)
    lora_metrics = evaluate(lora_model, loader, args)
    release(lora_model)

    base_loss = base_metrics["token_weighted_loss"]
    lora_loss = lora_metrics["token_weighted_loss"]
    base_ppl = base_metrics["perplexity"]
    lora_ppl = lora_metrics["perplexity"]
    report = {
        "schema_version": 1,
        "eval_data": args.eval_data,
        "base_weight_path": args.base_weight_path,
        "adapter_path": args.adapter_path,
        "max_seq_len": args.max_seq_len,
        "materialization_seed": args.seed,
        "base": base_metrics,
        "lora": lora_metrics,
        "delta": {
            "loss_absolute": lora_loss - base_loss,
            "loss_relative_percent": (lora_loss / base_loss - 1.0) * 100.0,
            "perplexity_absolute": lora_ppl - base_ppl,
            "perplexity_relative_percent": (lora_ppl / base_ppl - 1.0) * 100.0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
