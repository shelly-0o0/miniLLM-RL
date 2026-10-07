"""Stage 2 Qwen3-4B QLoRA Agent-SFT entrypoint.

The trainer teaches the policy a behavior that online RL can optimize:
serialize a valid calculator call, consume the tool observation, terminate the
assistant turn, and emit a grounded final answer.  Only assistant tokens are
supervised; system, user, and tool-observation tokens remain conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from dataset.qwen_stage2 import (
    AssistantOnlyCollator,
    QwenAgentSFTDataset,
    audit_sft_dataset,
)
from trainer.qwen3_adapter import (
    load_policy_model,
    load_stage2_tokenizer,
    load_yaml_config,
    require_section,
    save_adapter_atomic,
    trainable_parameter_summary,
)


class JsonlTrainerCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return control
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": logs,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return control


DEFAULT_STRUCTURE_MARKERS = ("<tool_call>", "</tool_call>", "<|im_end|>")


def resolve_single_token_markers(tokenizer, markers=DEFAULT_STRUCTURE_MARKERS):
    """Resolve structural markers and reject tokenizers that split them.

    The repair loss is intentionally narrow: it increases supervision for the
    exact Qwen control tokens that open/close a tool call and terminate an
    assistant message.  Silently weighting only part of a split marker would
    change that objective, so fail early instead.
    """

    resolved = {}
    for marker in markers:
        encoded = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if len(encoded) != 1:
            raise ValueError(
                f"structure marker must be one tokenizer token: {marker!r} -> {encoded}"
            )
        resolved[marker] = int(encoded[0])
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"structure markers do not have distinct token ids: {resolved}")
    return resolved


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    structure_token_ids: set[int] | tuple[int, ...] | list[int],
    structure_token_weight: float,
) -> torch.Tensor:
    """Causal-LM cross entropy with extra weight on structural target tokens."""

    if structure_token_weight < 1.0:
        raise ValueError("structure_token_weight must be at least 1.0")
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, seq, vocab] and labels [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label batch/sequence dimensions must match")

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(logits.device)
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("batch contains no supervised causal-LM targets")

    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels)
    weights = valid.to(token_losses.dtype)
    if structure_token_weight > 1.0:
        structural = torch.zeros_like(valid)
        for token_id in structure_token_ids:
            structural.logical_or_(shift_labels.eq(int(token_id)))
        weights = torch.where(
            structural & valid,
            weights.new_full((), float(structure_token_weight)),
            weights,
        )
    return (token_losses * weights).sum() / weights.sum()


class StructuredTokenWeightedTrainer(Trainer):
    """Trainer whose SFT objective emphasizes agent-protocol boundaries."""

    def __init__(
        self,
        *args,
        structure_token_ids,
        structure_token_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # This loss is normalized within each micro-batch and deliberately
        # does not consume ``num_items_in_batch``. Transformers 4.57 uses this
        # flag to decide whether it must apply gradient-accumulation
        # normalization; leaving it True inflates reported loss and gradients
        # by the accumulation factor.
        self.model_accepts_loss_kwargs = False
        self.structure_token_ids = tuple(int(value) for value in structure_token_ids)
        self.structure_token_weight = float(structure_token_weight)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = weighted_causal_lm_loss(
            logits,
            labels,
            structure_token_ids=self.structure_token_ids,
            structure_token_weight=self.structure_token_weight,
        )
        return (loss, outputs) if return_outputs else loss


def snapshot_trainable_parameters(model):
    """Take a CPU snapshot used only by short smoke runs."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def summarize_trainable_change(model, before):
    changed = 0
    maximum_absolute_change = 0.0
    non_finite = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.isfinite(current).all():
            non_finite.append(name)
        difference = (current.float() - before[name].float()).abs()
        tensor_max = difference.max().item() if difference.numel() else 0.0
        if tensor_max > 0:
            changed += 1
            maximum_absolute_change = max(maximum_absolute_change, tensor_max)
    return {
        "tensor_count": len(before),
        "changed_tensors": changed,
        "maximum_absolute_change": maximum_absolute_change,
        "non_finite_tensors": non_finite,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-4B QLoRA Agent-SFT")
    parser.add_argument(
        "--config", default=str(ROOT / "configs/qwen3_4b/lora_sft.yaml")
    )
    parser.add_argument(
        "--resume", nargs="?", const=True, default=False,
        help="resume from the latest Trainer checkpoint or an explicit path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for an isolated smoke run.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Append a safe tag to all outputs so a smoke run cannot overwrite a full run.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Override initialization.adapter_path, primarily for an isolated smoke chain.",
    )
    parser.add_argument(
        "--structure-token-weight",
        type=float,
        default=None,
        help=(
            "Override training.structure_token_weight. Values above 1 emphasize "
            "tool-call boundaries and assistant termination tokens."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    model_config = require_section(config, "model")
    lora_config = require_section(config, "lora")
    initialization = dict(config.get("initialization") or {})
    data_config = require_section(config, "data")
    training = dict(require_section(config, "training"))
    output = dict(require_section(config, "output"))
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        training["max_steps"] = args.max_steps
    if args.structure_token_weight is not None:
        training["structure_token_weight"] = args.structure_token_weight
    structure_token_weight = float(training.get("structure_token_weight", 1.0))
    if structure_token_weight < 1.0:
        raise ValueError("structure_token_weight must be at least 1.0")
    if args.output_tag:
        if not args.output_tag.replace("-", "").replace("_", "").isalnum():
            raise ValueError("--output-tag may contain only letters, digits, '-' and '_'")
        tag = args.output_tag
        for key in ("trainer_dir", "adapter_dir"):
            path = Path(output[key])
            output[key] = str(path.with_name(f"{path.name}_{tag}"))
        metrics = Path(output["metrics_path"])
        output["metrics_path"] = str(
            metrics.with_name(f"{metrics.stem}_{tag}{metrics.suffix}")
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 QLoRA SFT requires a CUDA GPU")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    configured_adapter = args.adapter_path or initialization.get("adapter_path")
    configured_adapter = str(ROOT / configured_adapter) if configured_adapter else None
    expand_adapter_targets = bool(
        initialization.get("expand_target_modules", False)
    )
    if configured_adapter and not Path(configured_adapter).is_dir():
        raise FileNotFoundError(f"initial adapter does not exist: {configured_adapter}")
    tokenizer = load_stage2_tokenizer(
        configured_adapter or model_config["name"],
        cache_dir=model_config.get("cache_dir"),
        padding_side="right",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_im_end_as_eos=bool(model_config.get("use_im_end_as_eos", True)),
    )
    structure_token_ids = resolve_single_token_markers(tokenizer)
    dataset = QwenAgentSFTDataset(
        ROOT / data_config["train_path"],
        tokenizer,
        max_length=int(data_config.get("max_length", 1536)),
    )
    audit = audit_sft_dataset(
        dataset, samples=int(data_config.get("audit_samples", 128))
    )
    if audit["zero_supervision_count"]:
        raise RuntimeError(f"assistant-only mask audit failed: {audit}")

    model = load_policy_model(
        model_config,
        lora_config,
        device=training.get("device", "cuda:0"),
        adapter_path=configured_adapter,
        trainable=True,
        expand_adapter_targets=expand_adapter_targets,
    )
    adapter_expansion = getattr(model, "_stage2_adapter_expansion", None)
    parameter_summary = trainable_parameter_summary(model)
    smoke_snapshot = (
        snapshot_trainable_parameters(model) if args.max_steps is not None else None
    )
    print(json.dumps({"mask_audit": audit}, ensure_ascii=False))
    print(json.dumps({"parameters": parameter_summary}, ensure_ascii=False))
    if adapter_expansion:
        print(json.dumps({"adapter_expansion": adapter_expansion}, ensure_ascii=False))
    print(json.dumps({
        "structure_token_weight": structure_token_weight,
        "structure_token_ids": structure_token_ids,
    }, ensure_ascii=False))

    trainer_output = ROOT / output["trainer_dir"]
    adapter_output = ROOT / output["adapter_dir"]
    metrics_path = ROOT / output["metrics_path"]
    if adapter_output.exists() and not args.resume:
        raise FileExistsError(
            f"adapter output already exists: {adapter_output}; use --resume or a new --output-tag"
        )
    trainer_output.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, trainer_output / "used_config.yaml")
    (trainer_output / "runtime_overrides.json").write_text(
        json.dumps(
            {
                "max_steps": args.max_steps,
                "output_tag": args.output_tag,
                "initial_adapter": configured_adapter,
                "adapter_path_override": args.adapter_path,
                "expand_adapter_targets": expand_adapter_targets,
                "adapter_expansion": adapter_expansion,
                "structure_token_weight": structure_token_weight,
                "structure_token_ids": structure_token_ids,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (trainer_output / "mask_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    max_steps = int(training.get("max_steps", 0))
    training_args = TrainingArguments(
        output_dir=str(trainer_output),
        overwrite_output_dir=False,
        per_device_train_batch_size=int(training.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 16)
        ),
        num_train_epochs=float(training.get("epochs", 1)),
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training.get("logging_steps", 5)),
        logging_first_step=True,
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training.get("optim", "adamw_torch"),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("num_workers", 0)),
        ddp_find_unused_parameters=False,
        seed=seed,
        data_seed=seed,
    )
    trainer = StructuredTokenWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[JsonlTrainerCallback(metrics_path)],
        structure_token_ids=structure_token_ids.values(),
        structure_token_weight=structure_token_weight,
    )
    resume = args.resume
    if resume is True:
        resume = True
    elif isinstance(resume, str):
        resume = str(Path(resume).resolve())
    result = trainer.train(resume_from_checkpoint=resume or None)
    if trainer.is_world_process_zero():
        smoke_update = None
        if smoke_snapshot is not None:
            smoke_update = summarize_trainable_change(model, smoke_snapshot)
            if smoke_update["non_finite_tensors"]:
                raise RuntimeError(f"SFT smoke produced non-finite tensors: {smoke_update}")
            if smoke_update["changed_tensors"] == 0:
                raise RuntimeError(
                    "SFT smoke completed without changing a trainable tensor; "
                    "increase --max-steps or fix the learning-rate schedule"
                )
        save_adapter_atomic(model, tokenizer, adapter_output)
        final = {
            "seed": seed,
            "initial_adapter": configured_adapter,
            "dataset_rows": len(dataset),
            "mask_audit": audit,
            "parameters": parameter_summary,
            "expand_adapter_targets": expand_adapter_targets,
            "adapter_expansion": adapter_expansion,
            "structure_token_weight": structure_token_weight,
            "structure_token_ids": structure_token_ids,
            "train_metrics": result.metrics,
            "smoke_update": smoke_update,
            "adapter_dir": str(adapter_output),
        }
        (trainer_output / "final_summary.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
