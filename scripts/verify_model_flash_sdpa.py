"""Prove that MiniMind's actual attention module executes a Flash SDPA kernel."""

import argparse
import json
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model, setup_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="out/model_flash_sdpa_verification.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    setup_seed(42)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        flash_attn=True,
        max_position_embeddings=max(2048, args.seq_len),
    )
    model, _ = init_model(config, args.weight, tokenizer_path="model", save_dir="out", device=args.device)
    model.eval()
    inputs = torch.randint(3, config.vocab_size, (1, args.seq_len), device=args.device)
    error = None
    names = []
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = model(inputs)
        torch.cuda.synchronize()
        names = sorted({event.key for event in profiler.key_averages()})
        logits_finite = bool(torch.isfinite(output.logits).all())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logits_finite = False
    relevant = [
        name for name in names
        if "flash" in name.casefold() or "attention" in name.casefold() or "scaled_dot" in name.casefold()
    ]
    report = {
        "weight": args.weight,
        "model_shape": {"hidden_size": args.hidden_size, "layers": args.num_hidden_layers, "sequence_length": args.seq_len},
        "dtype": "bfloat16 autocast",
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "model_attention_flash_flag": all(layer.self_attn.flash for layer in model.model.layers),
        "forced_flash_forward_succeeded": error is None,
        "logits_finite": logits_finite,
        "profile_contains_flash_kernel": any("flash" in name.casefold() for name in names),
        "profile_kernel_names": relevant,
        "error": error,
        "scope": "Actual MiniMind full forward at the recorded shape, with PyTorch Flash SDPA forced; not a FlashAttention-2 package claim.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all((report["model_attention_flash_flag"], report["forced_flash_forward_succeeded"], report["logits_finite"], report["profile_contains_flash_kernel"])):
        raise SystemExit("Actual-model Flash SDPA verification failed")


if __name__ == "__main__":
    main()
