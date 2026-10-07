"""Compare teacher-forced protocol-token likelihood before and after SFT repair."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from dataset.gsm8k import sha256_file
from dataset.qwen_stage2 import QwenAgentSFTDataset
from trainer.qwen3_adapter import (
    load_policy_model,
    load_stage2_tokenizer,
    load_yaml_config,
    require_section,
)
from trainer.train_qwen_lora_sft import resolve_single_token_markers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teacher-forced Qwen Agent-SFT structure-token audit"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/qwen3_4b/track2/agent_sft_a.yaml"),
    )
    parser.add_argument(
        "--before-adapter",
        default="out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter",
    )
    parser.add_argument(
        "--after-adapter",
        default=(
            "out/stage2_track2/qwen3_4b/"
            "agent_sft_a_s42_adapter_structure_w8"
        ),
    )
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="out/run_meta/track2_qwen_sft_structure_w8_comparison.json",
    )
    return parser.parse_args()


def adapter_fingerprint(path: Path) -> dict:
    model_file = path / "adapter_model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing adapter weights: {model_file}")
    return {
        "path": str(path),
        "adapter_model_sha256": sha256_file(model_file),
    }


def evaluate_adapter(
    adapter_path: Path,
    dataset,
    marker_ids: dict[str, int],
    model_config: dict,
    lora_config: dict,
    *,
    device: str,
    limit: int,
) -> dict:
    model = load_policy_model(
        model_config,
        lora_config,
        device=device,
        adapter_path=str(adapter_path),
        trainable=False,
    ).eval()
    totals = {
        "assistant_nll": 0.0,
        "assistant_correct": 0,
        "assistant_top20": 0,
        "assistant_tokens": 0,
        "ordinary_nll": 0.0,
        "ordinary_correct": 0,
        "ordinary_top20": 0,
        "ordinary_structure_top1": 0,
        "ordinary_structure_top20": 0,
        "ordinary_tokens": 0,
        "structure_nll": 0.0,
        "structure_correct": 0,
        "structure_top20": 0,
        "structure_tokens": 0,
    }
    per_marker = {
        marker: {"nll": 0.0, "correct": 0, "top20": 0, "tokens": 0}
        for marker in marker_ids
    }
    dtype = getattr(torch, model_config.get("compute_dtype", "bfloat16"))
    marker_tensor = torch.tensor(list(marker_ids.values()), device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        for index in range(limit):
            row = dataset[index]
            input_ids = row["input_ids"].unsqueeze(0).to(device)
            attention_mask = row["attention_mask"].unsqueeze(0).to(device)
            labels = row["labels"].unsqueeze(0).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            losses = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape_as(shift_labels)
            predictions = shift_logits.argmax(dim=-1)
            top20_predictions = shift_logits.topk(
                k=min(20, shift_logits.shape[-1]), dim=-1
            ).indices
            structure = torch.zeros_like(valid)
            for marker, token_id in marker_ids.items():
                selected = valid & shift_labels.eq(token_id)
                per_marker[marker]["nll"] += losses[selected].sum().item()
                per_marker[marker]["correct"] += int(
                    predictions[selected].eq(token_id).sum().item()
                )
                per_marker[marker]["top20"] += int(
                    top20_predictions[selected].eq(token_id).any(dim=-1).sum().item()
                )
                per_marker[marker]["tokens"] += int(selected.sum().item())
                structure.logical_or_(selected)
            ordinary = valid & ~structure
            totals["ordinary_structure_top1"] += int(
                predictions[ordinary]
                .unsqueeze(-1)
                .eq(marker_tensor)
                .any(dim=-1)
                .sum()
                .item()
            )
            totals["ordinary_structure_top20"] += int(
                top20_predictions[ordinary]
                .unsqueeze(-1)
                .eq(marker_tensor)
                .any(dim=-1)
                .any(dim=-1)
                .sum()
                .item()
            )
            for name, selected in (
                ("assistant", valid),
                ("ordinary", ordinary),
                ("structure", structure),
            ):
                totals[f"{name}_nll"] += losses[selected].sum().item()
                totals[f"{name}_correct"] += int(
                    predictions[selected].eq(shift_labels[selected]).sum().item()
                )
                totals[f"{name}_top20"] += int(
                    top20_predictions[selected]
                    .eq(shift_labels[selected].unsqueeze(-1))
                    .any(dim=-1)
                    .sum()
                    .item()
                )
                totals[f"{name}_tokens"] += int(selected.sum().item())

    def finalize(prefix, values):
        count = values[f"{prefix}_tokens"]
        if count <= 0:
            raise RuntimeError(f"audit found no {prefix} targets")
        return {
            "tokens": count,
            "mean_nll": values[f"{prefix}_nll"] / count,
            "top1_accuracy": values[f"{prefix}_correct"] / count,
            "top20_accuracy": values[f"{prefix}_top20"] / count,
        }

    result = {
        **adapter_fingerprint(adapter_path),
        "rows": limit,
        "assistant": finalize("assistant", totals),
        "ordinary": finalize("ordinary", totals),
        "structure": finalize("structure", totals),
        "per_marker": {},
    }
    result["ordinary"]["structure_marker_top1_false_positive_rate"] = (
        totals["ordinary_structure_top1"] / totals["ordinary_tokens"]
    )
    result["ordinary"]["structure_marker_top20_candidate_rate"] = (
        totals["ordinary_structure_top20"] / totals["ordinary_tokens"]
    )
    for marker, values in per_marker.items():
        if values["tokens"] <= 0:
            raise RuntimeError(f"audit found no targets for marker {marker!r}")
        result["per_marker"][marker] = {
            "token_id": marker_ids[marker],
            "tokens": values["tokens"],
            "mean_nll": values["nll"] / values["tokens"],
            "top1_accuracy": values["correct"] / values["tokens"],
            "top20_accuracy": values["top20"] / values["tokens"],
        }
    numeric = [
        section[key]
        for section in (
            result["assistant"], result["ordinary"], result["structure"],
            *result["per_marker"].values(),
        )
        for key in ("mean_nll", "top1_accuracy", "top20_accuracy")
    ]
    numeric.extend([
        result["ordinary"]["structure_marker_top1_false_positive_rate"],
        result["ordinary"]["structure_marker_top20_candidate_rate"],
    ])
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f"non-finite structure audit result: {result}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("structure audit requires CUDA")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    config = load_yaml_config(Path(args.config))
    model_config = require_section(config, "model")
    lora_config = require_section(config, "lora")
    data_config = require_section(config, "data")
    before_path = (ROOT / args.before_adapter).resolve()
    after_path = (ROOT / args.after_adapter).resolve()
    tokenizer = load_stage2_tokenizer(
        str(before_path),
        cache_dir=model_config.get("cache_dir"),
        padding_side="right",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_im_end_as_eos=bool(model_config.get("use_im_end_as_eos", True)),
    )
    marker_ids = resolve_single_token_markers(tokenizer)
    dataset = QwenAgentSFTDataset(
        ROOT / data_config["train_path"],
        tokenizer,
        max_length=int(data_config.get("max_length", 1536)),
    )
    limit = min(args.limit, len(dataset))
    before = evaluate_adapter(
        before_path, dataset, marker_ids, model_config, lora_config,
        device=args.device, limit=limit,
    )
    after = evaluate_adapter(
        after_path, dataset, marker_ids, model_config, lora_config,
        device=args.device, limit=limit,
    )
    report = {
        "marker_ids": marker_ids,
        "before": before,
        "after": after,
        "delta_after_minus_before": {
            name: {
                "mean_nll": after[name]["mean_nll"] - before[name]["mean_nll"],
                "top1_accuracy": (
                    after[name]["top1_accuracy"] - before[name]["top1_accuracy"]
                ),
                "top20_accuracy": (
                    after[name]["top20_accuracy"] - before[name]["top20_accuracy"]
                ),
                **(
                    {
                        key: after[name][key] - before[name][key]
                        for key in (
                            "structure_marker_top1_false_positive_rate",
                            "structure_marker_top20_candidate_rate",
                        )
                    }
                    if name == "ordinary"
                    else {}
                ),
            }
            for name in ("assistant", "ordinary", "structure")
        },
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
