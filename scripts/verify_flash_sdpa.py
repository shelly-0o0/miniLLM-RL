"""Verify that the target GPU can execute PyTorch's Flash SDPA backend.

Calling ``scaled_dot_product_attention`` does not by itself prove that a flash
kernel was selected.  This script forces the backend, profiles the CUDA kernel
names, and compares it with the forced math backend on the same tensor shape.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile


def timed(q, k, v, backend, repeats):
    values = []
    peak = 0
    for index in range(repeats + 2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(q.device)
        torch.cuda.synchronize(q.device)
        started = time.perf_counter()
        with torch.inference_mode(), sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize(q.device)
        elapsed = time.perf_counter() - started
        if index >= 2:
            values.append(elapsed)
            peak = max(peak, torch.cuda.max_memory_allocated(q.device))
        del output
    return {"median_ms": statistics.median(values) * 1000, "peak_memory_mib": peak / 2**20}


def main():
    parser = argparse.ArgumentParser(description="Force and profile CUDA Flash SDPA")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="out/flash_sdpa_verification.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("CUDA is required for Flash SDPA verification")
    torch.manual_seed(42)
    device = torch.device(args.device)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)

    forced_flash_succeeded = False
    error = None
    kernel_names = []
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize(device)
        forced_flash_succeeded = True
        kernel_names = sorted({event.key for event in profiler.key_averages()})
    except Exception as exc:  # evidence is persisted rather than hidden
        error = f"{type(exc).__name__}: {exc}"

    flash_metrics = timed(q, k, v, SDPBackend.FLASH_ATTENTION, args.repeats) if forced_flash_succeeded else None
    math_metrics = timed(q, k, v, SDPBackend.MATH, max(3, args.repeats // 2))
    report = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "shape_b_h_s_d": list(shape),
        "flash_sdp_enabled_flag": torch.backends.cuda.flash_sdp_enabled(),
        "forced_flash_succeeded": forced_flash_succeeded,
        "profile_contains_flash_kernel": any("flash" in name.casefold() for name in kernel_names),
        "profile_kernel_names": [name for name in kernel_names if "attention" in name.casefold() or "flash" in name.casefold() or "scaled_dot" in name.casefold()],
        "forced_flash": flash_metrics,
        "forced_math": math_metrics,
        "flash_speedup_over_math": (
            math_metrics["median_ms"] / flash_metrics["median_ms"]
            if flash_metrics else None
        ),
        "error": error,
        "scope": "Primitive SDPA backend verification; this is not a claim of FlashAttention-2 package integration for every model shape.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not forced_flash_succeeded or not report["profile_contains_flash_kernel"]:
        raise SystemExit("Flash SDPA was not proven on this target")


if __name__ == "__main__":
    main()
