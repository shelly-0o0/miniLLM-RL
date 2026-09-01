"""Audit sparse-MoE routing load, entropy and gradient connectivity."""

import argparse
import json
import math
import os
import statistics
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model, setup_seed


def finite_gradients(named_parameters, marker):
    gradients = [parameter.grad for name, parameter in named_parameters if marker in name and parameter.grad is not None]
    return {
        "tensor_count": len(gradients),
        "all_finite": bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients),
        "global_l2_norm": math.sqrt(sum(float(gradient.float().pow(2).sum()) for gradient in gradients)),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit MiniMind Top-1 MoE routing")
    parser.add_argument("--weight", default="pretrain_partial")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="out/run_meta/moe_routing_audit.json")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tokenizer_path = os.path.join(repo_root, "model")
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(repo_root, args.save_dir)

    setup_seed(args.seed)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=True,
    )
    model, _ = init_model(
        config,
        args.weight,
        tokenizer_path=tokenizer_path,
        save_dir=save_dir,
        device=args.device,
    )
    model.train()
    captured = {}
    handles = []

    def hook(name):
        def save(_module, _inputs, output):
            captured[name] = output.detach().float().cpu()
        return save

    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            handles.append(module.register_forward_hook(hook(name)))

    input_ids = torch.randint(3, config.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    labels = input_ids.clone()
    result = model(input_ids, labels=labels)
    total_loss = result.loss + result.aux_loss
    total_loss.backward()

    layers = []
    aggregate_counts = torch.zeros(config.num_experts, dtype=torch.float64)
    for name, logits in sorted(captured.items()):
        probabilities = torch.softmax(logits, dim=-1)
        top1 = probabilities.argmax(dim=-1)
        counts = torch.bincount(top1, minlength=config.num_experts).to(torch.float64)
        ratios = counts / counts.sum().clamp(min=1)
        aggregate_counts += counts
        mean = ratios.mean().item()
        layers.append({
            "gate": name,
            "tokens": int(counts.sum().item()),
            "load_ratio": ratios.tolist(),
            "load_cv": float(ratios.std(unbiased=False).item() / max(mean, 1e-12)),
            "mean_router_entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).mean()),
        })
    aggregate = aggregate_counts / aggregate_counts.sum().clamp(min=1)
    aggregate_mean = aggregate.mean().item()
    named_parameters = list(model.named_parameters())
    report = {
        "weight": args.weight,
        "seed": args.seed,
        "input_shape": [args.batch_size, args.seq_len],
        "num_experts": config.num_experts,
        "top_k": config.num_experts_per_tok,
        "lm_loss": float(result.loss.detach()),
        "aux_loss": float(result.aux_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "layers": layers,
        "aggregate_load_ratio": aggregate.tolist(),
        "aggregate_load_cv": float(aggregate.std(unbiased=False).item() / max(aggregate_mean, 1e-12)),
        "mean_layer_entropy": statistics.fmean(layer["mean_router_entropy"] for layer in layers),
        "maximum_entropy": math.log(config.num_experts),
        "router_gradients": finite_gradients(named_parameters, ".mlp.gate."),
        "expert_gradients": finite_gradients(named_parameters, ".mlp.experts."),
        "scope": "One deterministic batch on a partially trained checkpoint; validates routing mechanics, not convergence or quality.",
    }
    for handle in handles:
        handle.remove()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not layers or not report["router_gradients"]["all_finite"] or not report["expert_gradients"]["all_finite"]:
        raise SystemExit("MoE routing audit failed")


if __name__ == "__main__":
    main()
