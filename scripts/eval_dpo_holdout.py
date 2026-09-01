"""Evaluate DPO preference margins on one persisted held-out pair set."""

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig
from trainer.train_dpo import logits_to_log_probs
from trainer.trainer_utils import init_model, setup_seed


def collect_margins(model, loader, device, autocast_ctx):
    margins = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            chosen_x = batch["x_chosen"].to(device)
            rejected_x = batch["x_rejected"].to(device)
            chosen_y = batch["y_chosen"].to(device)
            rejected_y = batch["y_rejected"].to(device)
            chosen_mask = batch["mask_chosen"].to(device)
            rejected_mask = batch["mask_rejected"].to(device)
            inputs = torch.cat([chosen_x, rejected_x])
            labels = torch.cat([chosen_y, rejected_y])
            masks = torch.cat([chosen_mask, rejected_mask])
            with autocast_ctx:
                logits = model(inputs).logits
                sequence_logps = (logits_to_log_probs(logits.float(), labels) * masks).sum(1)
            half = chosen_x.size(0)
            margins.append((sequence_logps[:half] - sequence_logps[half:]).float().cpu())
    return torch.cat(margins)


def summarize(name, margins, reference_margins, beta):
    implicit_margin = beta * (margins - reference_margins)
    return {
        "checkpoint": name,
        "pairs": margins.numel(),
        "chosen_preference_accuracy": (margins > 0).float().mean().item(),
        "mean_policy_pair_logprob_margin": margins.mean().item(),
        "mean_implicit_reward_margin": implicit_margin.mean().item(),
        "positive_implicit_reward_margin_rate": (implicit_margin > 0).float().mean().item(),
        "dpo_loss_against_reference": (-F.logsigmoid(implicit_margin)).mean().item(),
        "implicit_reward_margin_std": implicit_margin.std(unbiased=False).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind DPO on held-out pairs")
    parser.add_argument("--data_path", default="dataset/dpo_eval.jsonl")
    parser.add_argument("--reference_weight", default="full_sft")
    parser.add_argument("--policy_weight", default="dpo_holdout")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", default="out/eval/dpo_holdout_metrics.json")
    args = parser.parse_args()
    setup_seed(args.seed)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    reference, tokenizer = init_model(config, args.reference_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DPODataset applies stochastic empty-think post-processing.  Materialize
    # it once so the reference and trained policies see byte-identical token
    # pairs; otherwise an apparent improvement can come from different inputs.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    materialized = [dataset[index] for index in range(min(args.limit, len(dataset)))]
    loader = DataLoader(materialized, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda") else nullcontext()
    reference_margins = collect_margins(reference, loader, args.device, autocast_ctx)
    base = summarize(args.reference_weight, reference_margins, reference_margins, args.beta)
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy, _ = init_model(config, args.policy_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    policy_margins = collect_margins(policy, loader, args.device, autocast_ctx)
    trained = summarize(args.policy_weight, policy_margins, reference_margins, args.beta)
    result = {
        "data_path": os.path.abspath(args.data_path),
        "beta": args.beta,
        "seed": args.seed,
        "base": base,
        "dpo": trained,
        "changes": {
            "chosen_preference_accuracy_points": 100 * (trained["chosen_preference_accuracy"] - base["chosen_preference_accuracy"]),
            "dpo_loss_percent": 100 * (trained["dpo_loss_against_reference"] / base["dpo_loss_against_reference"] - 1),
            "mean_implicit_reward_margin": trained["mean_implicit_reward_margin"],
        },
    }
    if not all(math.isfinite(float(value)) for section in (base, trained) for value in section.values() if isinstance(value, (int, float))):
        raise RuntimeError("non-finite DPO metric")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
