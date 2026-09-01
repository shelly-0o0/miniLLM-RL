"""Evaluate teacher, student initialization, CE control and KD student."""

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
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def materialize(path, tokenizer, max_length, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = SFTDataset(path, tokenizer, max_length=max_length)
    rows = [dataset[index] for index in range(len(dataset))]
    input_ids, labels = zip(*rows)
    return TensorDataset(torch.stack(input_ids), torch.stack(labels))


def load_model(path, hidden_size, num_layers, device, dtype):
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return model.eval().to(device=device, dtype=dtype), parameters


@torch.inference_mode()
def evaluate(model, loader, device):
    total_nll = 0.0
    total_tokens = 0
    samples = 0
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tokens = int(labels[..., 1:].ne(-100).sum().item())
        if not tokens:
            continue
        result = model(input_ids, labels=labels)
        total_nll += float(result.loss.float().item()) * tokens
        total_tokens += tokens
        samples += input_ids.size(0)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
        peak = torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
    else:
        peak = 0.0
    elapsed = time.perf_counter() - started
    loss = total_nll / max(total_tokens, 1)
    return {
        "samples": samples,
        "assistant_tokens": total_tokens,
        "token_weighted_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "elapsed_seconds": elapsed,
        "assistant_tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "peak_memory_mib": peak,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate controlled MiniMind KD ablation")
    parser.add_argument("--eval_data", default="dataset/kd_rlaif_eval_2000.jsonl")
    parser.add_argument("--teacher", default="out/full_sft_768.pth")
    parser.add_argument("--student_init", default="out/full_sft_student_512.pth")
    parser.add_argument("--ce_student", default="out/full_ce_student_512.pth")
    parser.add_argument("--kd_student", default="out/full_dist_student_512.pth")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--output", default="out/eval/kd_comparison.json")
    parser.add_argument("--teacher_hidden_size", type=int, default=768)
    parser.add_argument("--student_hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = materialize(args.eval_data, tokenizer, args.max_seq_len, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    specs = {
        "teacher_768": (args.teacher, args.teacher_hidden_size),
        "student_init_512": (args.student_init, args.student_hidden_size),
        "ce_student_512": (args.ce_student, args.student_hidden_size),
        "kd_student_512": (args.kd_student, args.student_hidden_size),
    }
    metrics = {}
    parameter_counts = {}
    file_sizes = {}
    for name, (path, hidden_size) in specs.items():
        model, parameters = load_model(
            path, hidden_size, args.num_hidden_layers, args.device, dtype
        )
        metrics[name] = evaluate(model, loader, args.device)
        parameter_counts[name] = parameters
        file_sizes[name] = Path(path).stat().st_size
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ce = metrics["ce_student_512"]
    kd = metrics["kd_student_512"]
    teacher_params = parameter_counts["teacher_768"]
    student_params = parameter_counts["kd_student_512"]
    report = {
        "schema_version": 1,
        "eval_data": args.eval_data,
        "materialization_seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "metrics": metrics,
        "parameter_counts": parameter_counts,
        "checkpoint_bytes": file_sizes,
        "compression": {
            "student_over_teacher_parameter_ratio": student_params / teacher_params,
            "parameter_reduction_percent": (1.0 - student_params / teacher_params) * 100.0,
        },
        "kd_vs_ce": {
            "loss_absolute": kd["token_weighted_loss"] - ce["token_weighted_loss"],
            "loss_relative_percent": (
                kd["token_weighted_loss"] / ce["token_weighted_loss"] - 1.0
            ) * 100.0,
            "perplexity_absolute": kd["perplexity"] - ce["perplexity"],
            "perplexity_relative_percent": (
                kd["perplexity"] / ce["perplexity"] - 1.0
            ) * 100.0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
