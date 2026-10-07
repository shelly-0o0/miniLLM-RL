"""Stage 2 Qwen3 QLoRA Agentic GRPO.

Both Pure-GRPO and SFT->GRPO use this entrypoint.  The only intended
difference is ``initialization.adapter_path`` in the YAML config.  Rollouts,
calculator execution, strict verifier, action masks and GRPO math are shared
with Stage 1 so the scale-transfer comparison does not silently change the
environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.lm_dataset import AgentRLDataset
from trainer.agent_chat import render_agent_chat
from trainer.policy_optimization import group_relative_advantages, positive_kl_estimate
from trainer.qwen3_adapter import (
    load_policy_model,
    load_policy_reference_pair,
    load_stage2_tokenizer,
    load_yaml_config,
    model_device,
    require_section,
    save_adapter_atomic,
    selected_action_logps,
    trainable_parameter_summary,
)
from trainer.qwen_grpo_objective import trajectory_grpo_loss
from trainer.rollout_engine import TorchRolloutEngine, legal_action_blocked_ids
from trainer.train_agent import calculate_rewards, parse_tool_calls, rollout_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Stage 2 Agentic GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-candidate-groups",
        type=int,
        default=None,
        help="Override the candidate-group budget, primarily for timed pilots.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Append a safe tag to every output path so a pilot cannot overwrite a full run.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Override initialization.adapter_path, primarily for an isolated smoke chain.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("strict", "shaped"),
        default=None,
        help=(
            "Override training.reward_mode. Strict uses verifiable task success +/-1; "
            "shaped is reserved for measured sparse-reward curriculum pilots."
        ),
    )
    parser.add_argument(
        "--policy-device",
        default=None,
        help="Override training.policy_device (for measured placement pilots).",
    )
    parser.add_argument(
        "--reference-device",
        default=None,
        help="Override training.reference_device; may equal the policy device.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_state_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def collate(rows):
    return {
        "messages": [row["messages"] for row in rows],
        "tools": [row["tools"] for row in rows],
        "required_tools": [row["required_tools"] for row in rows],
        "gt": [row["gt"] for row in rows],
    }


def validate_on_policy_sampling(rollout_config: dict) -> None:
    """Require generation to sample from the distribution used by log-probs.

    The probability ledger recomputes the model's untempered full softmax.
    Temperature scaling or top-k/top-p truncation would create a different
    behavior policy, invalidating importance ratios and sampled k3 KL.
    """

    sampling = (
        float(rollout_config.get("temperature", 1.0)),
        int(rollout_config.get("top_k", 0)),
        float(rollout_config.get("top_p", 1.0)),
    )
    if sampling != (1.0, 0, 1.0):
        raise ValueError(
            "Stage 2 GRPO requires on-policy full-softmax sampling: "
            "temperature=1.0, top_k=0, top_p=1.0; "
            f"received {sampling}"
        )


def group_k3_diagnostics(
    policy_logps: list[torch.Tensor], reference_logps: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """Return the exact token-global k3 used by the streaming objective."""

    if not policy_logps or len(policy_logps) != len(reference_logps):
        raise ValueError("policy/reference trajectory lists must be non-empty and match")
    if any(policy.shape != reference.shape or not policy.numel()
           for policy, reference in zip(policy_logps, reference_logps)):
        raise ValueError("every policy/reference trajectory must be non-empty and match")
    total_tokens = sum(values.numel() for values in policy_logps)
    per_trajectory = [
        positive_kl_estimate(policy, reference).mean()
        for policy, reference in zip(policy_logps, reference_logps)
    ]
    group_mean = sum(
        positive_kl_estimate(policy, reference).sum()
        for policy, reference in zip(policy_logps, reference_logps)
    ) / total_tokens
    return group_mean, per_trajectory, total_tokens


def resolve_kl_gate_policy(training: dict) -> tuple[str, int, int]:
    """Validate how finite, over-threshold KL groups are handled.

    ``raise`` is the conservative default for smoke tests. Long formal runs
    may use bounded ``reject_group``: the unsafe group is consumed without a
    backward pass, its RNG/progress is checkpointed, and repeated failures
    still stop the run.
    """

    action = str(training.get("kl_gate_action", "raise"))
    if action not in {"raise", "reject_group"}:
        raise ValueError("training.kl_gate_action must be 'raise' or 'reject_group'")
    max_rejections = int(training.get("max_kl_rejections", 0))
    max_consecutive = int(training.get("max_consecutive_kl_rejections", 0))
    if action == "reject_group" and (max_rejections < 1 or max_consecutive < 1):
        raise ValueError(
            "reject_group requires positive max_kl_rejections and "
            "max_consecutive_kl_rejections"
        )
    if action == "reject_group" and int(training.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError(
            "reject_group requires gradient_accumulation_steps=1 so rejecting a "
            "group cannot discard another accepted group's accumulated gradient"
        )
    return action, max_rejections, max_consecutive


def pack_trajectory(prompt_ids, response_ids, response_mask, rollout_logps, device):
    ids = prompt_ids + response_ids
    full_response_mask = [0] * len(prompt_ids) + response_mask
    action_mask = torch.tensor(
        [full_response_mask[1:]], device=device, dtype=torch.float32
    )
    old_storage = [0.0] * max(len(prompt_ids) - 1, 0) + rollout_logps
    old_storage = torch.tensor(old_storage, device=device, dtype=torch.float32)
    selected_rollout = old_storage[action_mask[0].bool()]
    return {
        "input_ids": torch.tensor([ids], device=device, dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), device=device, dtype=torch.long),
        "action_mask": action_mask,
        "rollout_logps": selected_rollout,
    }


def no_grad_selected(model, packed, blocked_ids):
    device = model_device(model)
    with torch.no_grad():
        return selected_action_logps(
            model,
            packed["input_ids"].to(device),
            packed["attention_mask"].to(device),
            packed["action_mask"].to(device),
            blocked_token_ids=blocked_ids,
        ).float()


def train_without_dropout(model):
    """Keep training/checkpointing active while making policy log-probs stable."""

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    return model


def write_kl_failure_diagnostic(
    path: Path,
    *,
    tokenizer,
    candidate_group: int,
    trajectory_index: int,
    packed: dict,
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    completion: str,
    turns: list[str],
    trace: dict,
    reward: float,
    advantage: float,
    threshold: float,
    group_mean_kl_k3: float | None = None,
):
    """Persist enough evidence to diagnose a rejected or fatal KL group.

    The caller decides whether the candidate position is checkpointed after
    writing this diagnostic. No optimizer step has occurred at this point.
    """

    current = current_logps.detach().float().cpu()
    old = old_logps.detach().float().cpu()
    reference = reference_logps.detach().float().cpu()
    delta = (reference - current).clamp(min=-20.0, max=20.0)
    per_token_kl = torch.exp(delta) - delta - 1.0
    targets = packed["input_ids"][:, 1:][packed["action_mask"].bool()].cpu()
    largest = torch.topk(per_token_kl, k=min(12, per_token_kl.numel()))
    outliers = []
    for value, action_index in zip(largest.values.tolist(), largest.indices.tolist()):
        token_id = int(targets[action_index].item())
        outliers.append(
            {
                "action_index": int(action_index),
                "token_id": token_id,
                "token_text": tokenizer.decode(
                    [token_id], skip_special_tokens=False
                ),
                "current_logp": float(current[action_index]),
                "old_logp": float(old[action_index]),
                "reference_logp": float(reference[action_index]),
                "reference_minus_current": float(reference[action_index] - current[action_index]),
                "k3": float(value),
            }
        )
    payload = {
        "candidate_group": candidate_group,
        "trajectory_index": trajectory_index,
        "threshold": threshold,
        "gate_scope": "group_action_token_mean",
        "group_mean_kl_k3": group_mean_kl_k3,
        "action_tokens": int(current.numel()),
        "mean_kl_k3": float(per_token_kl.mean()),
        "max_kl_k3": float(per_token_kl.max()),
        "max_abs_current_minus_old": float((current - old).abs().max()),
        "reward": reward,
        "advantage": advantage,
        "outlier_tokens": outliers,
        "completion": completion,
        "turn_outputs": turns,
        "trace": trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    cli = parse_args()
    config_path = Path(cli.config).resolve()
    config = load_yaml_config(config_path)
    model_config = require_section(config, "model")
    lora_config = require_section(config, "lora")
    initialization = require_section(config, "initialization")
    data_config = require_section(config, "data")
    rollout_config = require_section(config, "rollout")
    training = dict(require_section(config, "training"))
    output = dict(require_section(config, "output"))
    reward_mode = cli.reward_mode or str(training.get("reward_mode", "strict"))
    if reward_mode not in {"strict", "shaped"}:
        raise ValueError(f"unsupported training.reward_mode={reward_mode!r}")
    validate_on_policy_sampling(rollout_config)
    kl_gate_action, max_kl_rejections, max_consecutive_kl_rejections = (
        resolve_kl_gate_policy(training)
    )
    if cli.max_candidate_groups is not None:
        if cli.max_candidate_groups < 1:
            raise ValueError("--max-candidate-groups must be positive")
        training["max_candidate_groups"] = cli.max_candidate_groups
    if cli.policy_device:
        training["policy_device"] = cli.policy_device
    if cli.reference_device:
        training["reference_device"] = cli.reference_device
    if cli.output_tag:
        if not cli.output_tag.replace("-", "").replace("_", "").isalnum():
            raise ValueError("--output-tag may contain only letters, digits, '-' and '_'")
        tag = cli.output_tag
        output["run_name"] = f'{output.get("run_name", "run")}_{tag}'
        for key in ("checkpoint_dir", "adapter_dir"):
            path = Path(output[key])
            output[key] = str(path.with_name(f"{path.name}_{tag}"))
        metrics = Path(output["metrics_path"])
        output["metrics_path"] = str(
            metrics.with_name(f"{metrics.stem}_{tag}{metrics.suffix}")
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 GRPO requires CUDA")
    if int(training.get("batch_size", 1)) != 1:
        raise ValueError("Stage 2 currently requires one prompt group per batch")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy_device = training.get("policy_device", "cuda:0")
    reference_device = training.get("reference_device", policy_device)
    checkpoint_dir = ROOT / output["checkpoint_dir"]
    adapter_dir = checkpoint_dir / "policy_adapter"
    reference_adapter_dir = checkpoint_dir / "reference_adapter"
    state_path = checkpoint_dir / "training_state.pt"
    final_adapter_dir = ROOT / output["adapter_dir"]
    metrics_path = ROOT / output["metrics_path"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_config_path = checkpoint_dir / "used_config.yaml"
    initial_runtime_path = checkpoint_dir / "runtime_overrides.json"
    if cli.resume:
        if not (
            state_path.exists()
            and adapter_dir.exists()
            and reference_adapter_dir.exists()
            and initial_config_path.exists()
            and initial_runtime_path.exists()
        ):
            raise FileNotFoundError("resume requested but Stage 2 checkpoint is incomplete")
    elif state_path.exists() or adapter_dir.exists() or initial_config_path.exists():
        raise FileExistsError(
            f"run directory already contains a checkpoint: {checkpoint_dir}; use --resume"
        )

    runtime_record = {
        "max_candidate_groups": cli.max_candidate_groups,
        "output_tag": cli.output_tag,
        "adapter_path_override": cli.adapter_path,
        "reward_mode_override": cli.reward_mode,
        "resolved_reward_mode": reward_mode,
        "policy_device_override": cli.policy_device,
        "reference_device_override": cli.reference_device,
        "resolved_policy_device": policy_device,
        "resolved_reference_device": reference_device,
    }
    if not cli.resume:
        shutil.copy2(config_path, initial_config_path)
        initial_runtime_path.write_text(
            json.dumps(runtime_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    segment_log = checkpoint_dir / "run_segments.jsonl"
    segment_index = (
        sum(1 for line in segment_log.open(encoding="utf-8") if line.strip())
        if segment_log.exists() else 0
    )
    segment_config_path = checkpoint_dir / f"segment_{segment_index:03d}_config.yaml"
    shutil.copy2(config_path, segment_config_path)
    write_jsonl(
        segment_log,
        {
            "segment": segment_index,
            "resume": bool(cli.resume),
            "config_snapshot": str(segment_config_path.relative_to(ROOT)),
            **runtime_record,
        },
    )

    configured_adapter = cli.adapter_path or initialization.get("adapter_path")
    configured_adapter = str(ROOT / configured_adapter) if configured_adapter else None
    if cli.resume:
        policy = load_policy_model(
            model_config, lora_config, device=policy_device,
            adapter_path=str(adapter_dir), trainable=True,
        )
        reference = load_policy_model(
            model_config, lora_config, device=reference_device,
            adapter_path=str(reference_adapter_dir), trainable=False,
        ).eval().requires_grad_(False)
    else:
        policy, reference = load_policy_reference_pair(
            model_config, lora_config,
            policy_device=policy_device,
            reference_device=reference_device,
            adapter_path=configured_adapter,
        )
    # Gradient checkpointing in Transformers requires train mode.  Dropout,
    # however, would make current log-probs incomparable with eval-mode
    # behavior/reference log-probs, so disable only Dropout submodules.
    train_without_dropout(policy)

    tokenizer_source = configured_adapter or model_config["name"]
    tokenizer = load_stage2_tokenizer(
        tokenizer_source,
        cache_dir=model_config.get("cache_dir"),
        padding_side="left",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_im_end_as_eos=bool(model_config.get("use_im_end_as_eos", True)),
    )
    if not cli.resume:
        save_adapter_atomic(reference, tokenizer, reference_adapter_dir)
        save_adapter_atomic(policy, tokenizer, adapter_dir)

    summary = trainable_parameter_summary(policy)
    print(json.dumps({"parameters": summary}, ensure_ascii=False))
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("policy has no trainable LoRA parameters")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(training.get("learning_rate", 1e-6)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    dataset = AgentRLDataset(
        ROOT / data_config["train_path"], tokenizer,
        max_length=int(data_config.get("max_prompt_length", 1024)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    epochs = int(training.get("epochs", 1))
    max_candidate_groups = int(training.get("max_candidate_groups", len(dataset)))
    policy_update_epochs = int(training.get("policy_update_epochs", 1))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    estimated_updates = math.ceil(
        min(len(dataset), max_candidate_groups) * policy_update_epochs
        / accumulation_steps
    ) * epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(estimated_updates, 1),
        eta_min=float(training.get("learning_rate", 1e-6)) / 10,
    )
    start_epoch = 0
    start_candidate = 0
    optimizer_updates = 0
    kl_rejections = 0
    consecutive_kl_rejections = 0
    if cli.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        start_candidate = int(state["candidate_group"])
        optimizer_updates = int(state["optimizer_updates"])
        kl_rejections = int(state.get("kl_rejections", 0))
        consecutive_kl_rejections = int(
            state.get("consecutive_kl_rejections", 0)
        )
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])

    rollout_engine = TorchRolloutEngine(
        policy_model=policy,
        tokenizer=tokenizer,
        device=policy_device,
        autocast_ctx=torch.autocast(
            device_type="cuda",
            dtype=getattr(torch, model_config.get("compute_dtype", "bfloat16")),
        ),
    )
    blocked_ids = legal_action_blocked_ids(tokenizer)
    group_size = int(rollout_config.get("num_generations", 8))
    if group_size < 2:
        raise ValueError("GRPO num_generations must be at least 2")
    optimizer.zero_grad(set_to_none=True)
    optimization_micro_step = 0
    run_started = time.time()

    for epoch in range(start_epoch, epochs):
        for candidate_group, batch in enumerate(loader, start=1):
            if epoch == start_epoch and candidate_group <= start_candidate:
                continue
            if candidate_group > max_candidate_groups:
                break
            rollout_values = rollout_batch(
                rollout_engine,
                tokenizer,
                batch["messages"],
                batch["tools"],
                group_size,
                max_turns=int(rollout_config.get("max_turns", 3)),
                max_new_tokens=int(rollout_config.get("max_new_tokens", 384)),
                thinking_ratio=0.0,
                temperature=float(rollout_config.get("temperature", 0.7)),
                top_k=int(rollout_config.get("top_k", 20)),
                top_p=float(rollout_config.get("top_p", 0.95)),
                device=policy_device,
            )
            (
                completions, _, prompt_ids_batch, response_ids_batch,
                response_masks_batch, response_old_logps_batch,
                turn_outputs_batch, unfinished_batch, traces_batch,
            ) = rollout_values
            prompts = [
                render_agent_chat(
                    tokenizer, messages, tools=tools, tokenize=False,
                    add_generation_prompt=True, open_thinking=False,
                )
                for messages, tools in zip(batch["messages"], batch["tools"])
            ]
            reward_output = calculate_rewards(
                prompts,
                completions,
                batch["gt"],
                batch["tools"],
                group_size,
                batch["required_tools"],
                reward_model=None,
                device=policy_device,
                turn_outputs_batch=turn_outputs_batch,
                unfinished_batch=unfinished_batch,
                require_tool_call_for_success=True,
                reward_mode=reward_mode,
                return_details=True,
            )
            advantages = group_relative_advantages(
                reward_output.rewards, group_size
            ).detach()
            packed = [
                pack_trajectory(prompt_ids, response_ids, response_mask, old_logps, policy_device)
                for prompt_ids, response_ids, response_mask, old_logps in zip(
                    prompt_ids_batch, response_ids_batch,
                    response_masks_batch, response_old_logps_batch,
                )
            ]
            max_total_length = int(rollout_config.get("max_total_length", 2500))
            actual_max = max(item["input_ids"].size(1) for item in packed)
            if actual_max > max_total_length:
                raise RuntimeError(
                    f"trajectory length {actual_max} exceeds {max_total_length}; "
                    "post-rollout truncation would invalidate behavior log-probabilities"
                )

            old_logps = []
            reference_logps = []
            rollout_mae_values = []
            was_training = policy.training
            policy.eval()
            for item in packed:
                recomputed = no_grad_selected(policy, item, blocked_ids).to(policy_device)
                reference_values = no_grad_selected(reference, item, blocked_ids).to(policy_device)
                if recomputed.numel() != item["rollout_logps"].numel():
                    raise RuntimeError("action/log-probability ledger length mismatch")
                rollout_mae_values.append(
                    (recomputed - item["rollout_logps"]).abs().mean()
                )
                old_logps.append(recomputed.detach())
                reference_logps.append(reference_values.detach())
            policy.train(was_training)
            if was_training:
                train_without_dropout(policy)
            rollout_logprob_mae = torch.stack(rollout_mae_values).mean()
            threshold = float(training.get("max_rollout_logprob_mae", 0.1))
            if rollout_logprob_mae.item() > threshold:
                raise RuntimeError(
                    f"rollout log-probability MAE {rollout_logprob_mae.item():.6f} "
                    f"exceeds safety threshold {threshold:.6f}"
                )
            total_action_tokens = sum(value.numel() for value in old_logps)
            if not total_action_tokens:
                raise RuntimeError("rollout group contains no policy action tokens")

            # The objective normalizes KL over every action token in the
            # prompt group, so the safety gate must use that same estimator.
            # A per-trajectory gate can reject a valid group because one rare
            # sampled token gives k3 a large but diluted Monte-Carlo outlier.
            (
                preupdate_group_kl,
                trajectory_kl_means,
                checked_action_tokens,
            ) = group_k3_diagnostics(old_logps, reference_logps)
            if checked_action_tokens != total_action_tokens:
                raise RuntimeError("KL diagnostic/action-token ledger mismatch")
            max_group_kl = float(
                training.get(
                    "max_group_kl_k3",
                    training.get("max_trajectory_kl_k3", 10.0),
                )
            )
            group_kl_value = float(preupdate_group_kl.item())
            if not math.isfinite(group_kl_value) or group_kl_value > max_group_kl:
                optimizer.zero_grad(set_to_none=True)
                worst_index = max(
                    range(len(trajectory_kl_means)),
                    key=lambda value: trajectory_kl_means[value].item(),
                )
                can_reject = (
                    math.isfinite(group_kl_value)
                    and kl_gate_action == "reject_group"
                )
                if can_reject:
                    kl_rejections += 1
                    consecutive_kl_rejections += 1
                    diagnostic_path = (
                        checkpoint_dir
                        / "safety_rejections"
                        / f"candidate_group_{candidate_group:06d}.json"
                    )
                else:
                    diagnostic_path = checkpoint_dir / "safety_failure.json"
                write_kl_failure_diagnostic(
                    diagnostic_path,
                    tokenizer=tokenizer,
                    candidate_group=candidate_group,
                    trajectory_index=worst_index,
                    packed=packed[worst_index],
                    current_logps=old_logps[worst_index],
                    old_logps=old_logps[worst_index],
                    reference_logps=reference_logps[worst_index],
                    completion=completions[worst_index],
                    turns=turn_outputs_batch[worst_index],
                    trace=traces_batch[worst_index],
                    reward=float(reward_output.rewards[worst_index]),
                    advantage=float(advantages[worst_index]),
                    threshold=max_group_kl,
                    group_mean_kl_k3=group_kl_value,
                )
                limits_exceeded = (
                    kl_rejections > max_kl_rejections
                    or consecutive_kl_rejections > max_consecutive_kl_rejections
                )
                if not can_reject or limits_exceeded:
                    if diagnostic_path.name != "safety_failure.json":
                        shutil.copy2(
                            diagnostic_path, checkpoint_dir / "safety_failure.json"
                        )
                    raise RuntimeError(
                        "group KL safety gate failed before optimizer step: "
                        f"kl_k3={group_kl_value:.6f}, "
                        f"threshold={max_group_kl:.6f}, "
                        f"rejections={kl_rejections}/{max_kl_rejections}, "
                        "consecutive_rejections="
                        f"{consecutive_kl_rejections}/"
                        f"{max_consecutive_kl_rejections}; "
                        "check the saved token diagnostic"
                    )

                rejection_metrics = {
                    "epoch": epoch + 1,
                    "candidate_group": candidate_group,
                    "candidate_trajectories": candidate_group * group_size,
                    "optimizer_updates": optimizer_updates,
                    "reason": "group_kl_k3_exceeds_threshold",
                    "group_kl_k3": group_kl_value,
                    "threshold": max_group_kl,
                    "worst_trajectory_kl_k3": float(
                        trajectory_kl_means[worst_index]
                    ),
                    "rollout_logprob_mae": float(rollout_logprob_mae),
                    "kl_rejections": kl_rejections,
                    "consecutive_kl_rejections": consecutive_kl_rejections,
                    "diagnostic": str(diagnostic_path.relative_to(ROOT)),
                    "wall_time_seconds": time.time() - run_started,
                }
                print(
                    json.dumps(
                        {"safety_rejection": rejection_metrics}, ensure_ascii=False
                    ),
                    flush=True,
                )
                write_jsonl(
                    metrics_path,
                    {"split": "safety_rejection", "metrics": rejection_metrics},
                )
                save_adapter_atomic(policy, tokenizer, adapter_dir)
                save_state_atomic(
                    state_path,
                    {
                        "epoch": epoch,
                        "candidate_group": candidate_group,
                        "optimizer_updates": optimizer_updates,
                        "kl_rejections": kl_rejections,
                        "consecutive_kl_rejections": consecutive_kl_rejections,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                )
                rollout_engine.update_policy(policy)
                continue

            consecutive_kl_rejections = 0

            last_outputs = []
            gradient_norm = 0.0
            for _ in range(policy_update_epochs):
                last_outputs = []
                for index, item in enumerate(packed):
                    current = selected_action_logps(
                        policy,
                        item["input_ids"],
                        item["attention_mask"],
                        item["action_mask"],
                        blocked_token_ids=blocked_ids,
                    )
                    objective = trajectory_grpo_loss(
                        current,
                        old_logps[index],
                        reference_logps[index],
                        advantage=advantages[index],
                        group_size=group_size,
                        total_action_tokens=total_action_tokens,
                        beta=float(training.get("beta", 0.02)),
                        epsilon=float(training.get("epsilon", 0.2)),
                    )
                    (objective.loss / accumulation_steps).backward()
                    last_outputs.append(objective)
                optimization_micro_step += 1
                if optimization_micro_step % accumulation_steps == 0:
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(training.get("max_grad_norm", 1.0)),
                    ))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_updates += 1

            rewards = reward_output.rewards
            group_std = rewards.std(unbiased=False)
            metrics = {
                "epoch": epoch + 1,
                "reward_mode": reward_mode,
                "candidate_group": candidate_group,
                "candidate_trajectories": candidate_group * group_size,
                "optimizer_updates": optimizer_updates,
                "reward": rewards.mean().item(),
                "task_accuracy": reward_output.task_success.float().mean().item(),
                "answer_accuracy": reward_output.answer_accuracy.mean().item(),
                "format_valid_rate": reward_output.format_valid.mean().item(),
                "tool_call_valid_rate": reward_output.tool_call_valid.mean().item(),
                "tool_execution_success_rate": reward_output.tool_execution_success.mean().item(),
                "required_tool_coverage_rate": reward_output.required_tool_coverage.mean().item(),
                "tool_evidence_coverage_rate": reward_output.tool_evidence_coverage.mean().item(),
                "protocol_progress": reward_output.protocol_progress.mean().item(),
                "group_reward_std": group_std.item(),
                "zero_variance_group": float(group_std.item() < 1e-6),
                "advantages_std": advantages.std(unbiased=False).item(),
                "policy_loss": sum(value.policy_contribution.item() for value in last_outputs),
                "kl_loss": sum(value.kl_contribution.item() for value in last_outputs),
                "kl_k3": sum(
                    value.approx_kl.item() * value.token_count for value in last_outputs
                ) / total_action_tokens,
                "preupdate_group_kl_k3": preupdate_group_kl.item(),
                "gradient_norm_before_clip": gradient_norm,
                "clip_fraction": sum(
                    value.clip_fraction.item() * value.token_count for value in last_outputs
                ) / total_action_tokens,
                "ratio_mean": sum(
                    value.ratio_mean.item() * value.token_count for value in last_outputs
                ) / total_action_tokens,
                "rollout_logprob_mae": rollout_logprob_mae.item(),
                "action_tokens": total_action_tokens,
                "tool_calls": sum(
                    len(parse_tool_calls(turn))
                    for turns in turn_outputs_batch for turn in turns
                ),
                "unfinished_rate": sum(bool(value) for value in unfinished_batch) / group_size,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "wall_time_seconds": time.time() - run_started,
            }
            logging_steps = int(training.get("logging_steps", 1))
            if candidate_group % logging_steps == 0 or candidate_group == 1:
                print(json.dumps(metrics, ensure_ascii=False), flush=True)
                write_jsonl(metrics_path, {"split": "train", "metrics": metrics})

            save_steps = int(training.get("save_steps", 50))
            if candidate_group % save_steps == 0 or candidate_group == max_candidate_groups:
                save_adapter_atomic(policy, tokenizer, adapter_dir)
                save_state_atomic(
                    state_path,
                    {
                        "epoch": epoch,
                        "candidate_group": candidate_group,
                        "optimizer_updates": optimizer_updates,
                        "kl_rejections": kl_rejections,
                        "consecutive_kl_rejections": consecutive_kl_rejections,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                )
            rollout_engine.update_policy(policy)

        start_candidate = 0

    if optimization_micro_step % accumulation_steps:
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(training.get("max_grad_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_updates += 1
    save_adapter_atomic(policy, tokenizer, final_adapter_dir)
    final = {
        "run_name": output.get("run_name"),
        "seed": seed,
        "reward_mode": reward_mode,
        "initial_adapter": configured_adapter,
        "dataset_rows": len(dataset),
        "group_size": group_size,
        "candidate_groups": min(len(dataset), max_candidate_groups) * epochs,
        "candidate_trajectories": min(len(dataset), max_candidate_groups) * epochs * group_size,
        "optimizer_updates": optimizer_updates,
        "kl_rejections": kl_rejections,
        "segment_wall_time_seconds": time.time() - run_started,
        "adapter_dir": str(final_adapter_dir),
        "parameters": summary,
    }
    write_jsonl(metrics_path, {"split": "train", "final": final})
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
