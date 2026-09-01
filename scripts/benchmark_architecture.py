"""Measure MiniMind decode speed, peak memory and theoretical GQA KV cache."""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model, setup_seed


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def benchmark_generate(model, prompt, *, use_cache, decode_tokens, repeats, device):
    times, peaks = [], []
    for repeat in range(repeats + 1):
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        synchronize(device)
        started = time.perf_counter()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda") else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            output = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=decode_tokens,
                do_sample=False,
                use_cache=use_cache,
                eos_token_id=None,
            )
        synchronize(device)
        elapsed = time.perf_counter() - started
        peak = (torch.cuda.max_memory_allocated(torch.device(device))
                if device.startswith("cuda") else 0)
        if repeat:
            times.append(elapsed)
            peaks.append(peak)
        del output
    return {
        "seconds_median": statistics.median(times),
        "tokens_per_second": decode_tokens / statistics.median(times),
        "peak_memory_mib": max(peaks) / 2**20 if peaks else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniMind architecture mechanisms")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--save_dir", default="./out")
    parser.add_argument("--tokenizer_path", default="./model")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--flash_attn", type=int, default=1, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--prompt_length", type=int, default=1024)
    parser.add_argument("--decode_tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="./out/architecture_benchmark.csv")
    args = parser.parse_args()

    setup_seed(42)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=bool(args.use_moe),
        flash_attn=bool(args.flash_attn),
        inference_rope_scaling=bool(args.inference_rope_scaling),
        max_position_embeddings=args.max_position_embeddings,
    )
    model, tokenizer = init_model(
        config, args.weight, tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir, device=args.device,
    )
    model.eval()
    prompt = torch.randint(
        low=3, high=config.vocab_size,
        size=(1, args.prompt_length), device=args.device,
    )
    cached = benchmark_generate(
        model, prompt, use_cache=True, decode_tokens=args.decode_tokens,
        repeats=args.repeats, device=args.device,
    )
    uncached = benchmark_generate(
        model, prompt, use_cache=False, decode_tokens=args.decode_tokens,
        repeats=args.repeats, device=args.device,
    )
    dtype_bytes = 2 if args.device.startswith("cuda") else 4
    kv_cache_bytes = (
        2 * args.num_hidden_layers * (args.prompt_length + args.decode_tokens)
        * args.num_key_value_heads * config.head_dim * dtype_bytes
    )
    mha_cache_bytes = kv_cache_bytes * args.num_attention_heads / args.num_key_value_heads
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    expert_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if "mlp.experts.0." in name
    )
    active_parameters = total_parameters
    if args.use_moe:
        active_parameters = (
            total_parameters - expert_parameters * config.num_experts
            + expert_parameters * config.num_experts_per_tok
        )
    row = {
        "weight": args.weight,
        "device": args.device,
        "prompt_length": args.prompt_length,
        "decode_tokens": args.decode_tokens,
        "flash_attn": args.flash_attn,
        "yarn": args.inference_rope_scaling,
        "use_moe": args.use_moe,
        "total_parameters": total_parameters,
        "active_parameters_per_token_approx": active_parameters,
        "gqa_kv_cache_mib_theoretical": kv_cache_bytes / 2**20,
        "mha_kv_cache_mib_theoretical": mha_cache_bytes / 2**20,
        "gqa_cache_reduction_ratio": 1.0 - kv_cache_bytes / mha_cache_bytes,
        "cached_tokens_per_second": cached["tokens_per_second"],
        "uncached_tokens_per_second": uncached["tokens_per_second"],
        "cached_peak_memory_mib": cached["peak_memory_mib"],
        "uncached_peak_memory_mib": uncached["peak_memory_mib"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    exists = os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
