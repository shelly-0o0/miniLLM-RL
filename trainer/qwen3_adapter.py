"""Qwen3/PEFT model adapter used by the Stage 2 Agentic RLVR pipeline.

Stage 1 owns the environment and policy objectives; this module only adapts a
Hugging Face Qwen3 causal LM to those interfaces.  Imports that require the
Stage 2 environment are intentionally lazy so data/config tests remain usable
without downloading a 4B model or importing bitsandbytes.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Stage 2 config must be a mapping: {path}")
    return value


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Stage 2 config is missing mapping section {name!r}")
    return value


def resolve_dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if value not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported torch dtype: {name!r}")
    return value


def build_quantization_config(model_config: dict[str, Any]):
    if not model_config.get("load_in_4bit", True):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            model_config.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=resolve_dtype(
            model_config.get("compute_dtype", "bfloat16")
        ),
    )


def build_lora_config(lora_config: dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=lora_config.get("bias", "none"),
        target_modules=lora_config.get("target_modules", "all-linear"),
    )


def load_stage2_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: str | None = None,
    padding_side: str = "left",
    trust_remote_code: bool = False,
    use_im_end_as_eos: bool = True,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        # Transformers 4.57 detects the legacy pre-tokenizer regex shipped by
        # this tokenizer family.  Opting in here makes local and Hub loads use
        # the corrected token boundaries instead of merely emitting a warning.
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_im_end_as_eos:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if im_end is not None and im_end >= 0 and im_end != unknown:
            tokenizer.eos_token_id = int(im_end)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer {model_name_or_path!r} has no chat template; a Qwen3 "
            "tool-aware tokenizer is required"
        )
    return tokenizer


def _model_load_kwargs(model_config: dict[str, Any], device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    cache_dir = model_config.get("cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    revision = model_config.get("revision")
    if revision:
        kwargs["revision"] = revision
    quantization_config = build_quantization_config(model_config)
    dtype = resolve_dtype(model_config.get("compute_dtype", "bfloat16"))
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": device}
    else:
        kwargs["dtype"] = dtype
    return kwargs


def load_base_model(
    model_config: dict[str, Any],
    *,
    device: str,
    for_training: bool,
):
    from transformers import AutoModelForCausalLM

    model_name = model_config["name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **_model_load_kwargs(model_config, device)
    )
    if not model_config.get("load_in_4bit", True):
        model.to(device)
    model.config.use_cache = not for_training
    if for_training:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def load_policy_model(
    model_config: dict[str, Any],
    lora_config: dict[str, Any],
    *,
    device: str,
    adapter_path: str | None = None,
    trainable: bool = True,
    expand_adapter_targets: bool = False,
):
    from peft import (
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from peft.utils.save_and_load import load_peft_weights

    if expand_adapter_targets and (not adapter_path or not trainable):
        raise ValueError(
            "expand_adapter_targets requires a trainable source adapter"
        )

    base = load_base_model(model_config, device=device, for_training=trainable)
    if trainable and model_config.get("load_in_4bit", True):
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if adapter_path and expand_adapter_targets:
        # Recreate the adapter with the requested superset of target modules,
        # then copy every tensor from the old adapter. New LoRA-B tensors are
        # zero-initialized, so adding lm_head starts functionally identical to
        # the source checkpoint instead of discarding the completed SFT.
        model = get_peft_model(base, build_lora_config(lora_config))
        source_state = load_peft_weights(adapter_path, device="cpu")
        set_peft_model_state_dict(model, source_state)
        expanded_state = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(
                model, save_embedding_layers=False
            ).items()
        }
        missing_source = sorted(set(source_state) - set(expanded_state))
        if missing_source:
            raise RuntimeError(
                f"expanded adapter did not accept source tensors: {missing_source[:5]}"
            )
        changed_source = []
        for key, source in source_state.items():
            loaded = expanded_state[key]
            if not torch.equal(source.to(dtype=loaded.dtype), loaded):
                changed_source.append(key)
        if changed_source:
            raise RuntimeError(
                f"expanded adapter changed source tensors: {changed_source[:5]}"
            )
        added = sorted(set(expanded_state) - set(source_state))
        if not added or not all("lm_head" in key for key in added):
            raise RuntimeError(
                "adapter expansion must add only lm_head LoRA tensors; "
                f"added={added[:10]}"
            )
        model._stage2_adapter_expansion = {
            "source_tensor_count": len(source_state),
            "added_tensors": added,
        }
    elif adapter_path:
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=trainable
        )
    else:
        model = get_peft_model(base, build_lora_config(lora_config))
        if not trainable:
            model.requires_grad_(False)
    model.config.use_cache = not trainable
    return model


def load_policy_reference_pair(
    model_config: dict[str, Any],
    lora_config: dict[str, Any],
    *,
    policy_device: str,
    reference_device: str,
    adapter_path: str | None,
):
    """Load trainable policy and frozen reference from exactly one start state."""

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    policy = load_policy_model(
        model_config,
        lora_config,
        device=policy_device,
        adapter_path=adapter_path,
        trainable=True,
    )
    reference = load_policy_model(
        model_config,
        lora_config,
        device=reference_device,
        adapter_path=adapter_path,
        trainable=False,
    )
    # With a fresh Pure-GRPO adapter, both independently initialized LoRA
    # modules must still be made byte-identical.  This also protects against a
    # future PEFT initializer that is not zero-equivalent.
    state = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(policy).items()
    }
    set_peft_model_state_dict(reference, state)
    reference.eval().requires_grad_(False)
    return policy, reference


def trainable_parameter_summary(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / max(total, 1),
    }


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise ValueError("model has no materialized parameter device")


def selected_action_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    blocked_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Compute log-probabilities only at policy-action positions.

    Qwen3 accepts a tensor in ``logits_to_keep``.  Selecting the predecessor
    positions of action tokens avoids materializing ``sequence × 151k`` logits
    for every tool observation and prompt token, which is essential for a 4B
    policy on 24 GiB GPUs.  A compatibility fallback is retained for model
    versions without indexed logits.
    """

    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("selected_action_logps currently expects batch size 1")
    if action_mask.shape != (1, input_ids.size(1) - 1):
        raise ValueError("action_mask must have shape [1, sequence_length - 1]")
    positions = action_mask[0].bool().nonzero(as_tuple=False).flatten()
    if not positions.numel():
        return torch.empty(0, device=input_ids.device, dtype=torch.float32)
    targets = input_ids[:, 1:][:, positions]
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=positions,
        )
        logits = output.logits
        if logits.size(1) != positions.numel():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits[:, positions, :]
    except TypeError:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, positions, :]
    logits = logits.float().clone()
    for token_id in blocked_token_ids or []:
        if token_id is not None and 0 <= int(token_id) < logits.size(-1):
            logits[..., int(token_id)] = -torch.inf
    return torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[0]


def save_adapter_atomic(model, tokenizer, output_dir: str | Path) -> Path:
    """Atomically replace one PEFT adapter directory."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    previous = output_dir.with_name(output_dir.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    # No Stage 2 run resizes embeddings. Explicitly disabling PEFT's automatic
    # embedding save avoids serializing the full tied Qwen embedding/lm_head
    # matrix when lm_head itself is a LoRA target; only adapter deltas belong
    # in this artifact.
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(temporary)
    metadata = {
        "format": "peft_adapter",
        "base_model": getattr(model, "base_model_name_or_path", None),
    }
    (temporary / "stage2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if previous.exists():
        shutil.rmtree(previous)
    if output_dir.exists():
        os.replace(output_dir, previous)
    os.replace(temporary, output_dir)
    if previous.exists():
        shutil.rmtree(previous)
    return output_dir
