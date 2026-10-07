"""Memory-bounded GRPO loss pieces for Qwen3 Stage 2.

The canonical objective in :mod:`trainer.policy_optimization` operates on a
dense ``[group, sequence]`` tensor.  A 4B Qwen policy has a ~152k vocabulary,
so Stage 2 computes one trajectory at a time.  The scaling below makes the sum
of per-trajectory backward calls exactly equal to GRPO's sequence-first policy
mean plus a token-global KL mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trainer.policy_optimization import positive_kl_estimate


@dataclass
class TrajectoryGRPOOutput:
    loss: torch.Tensor
    policy_contribution: torch.Tensor
    kl_contribution: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_std: torch.Tensor
    approx_kl: torch.Tensor
    token_count: int


def trajectory_grpo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    *,
    advantage: torch.Tensor | float,
    group_size: int,
    total_action_tokens: int,
    beta: float,
    epsilon: float,
) -> TrajectoryGRPOOutput:
    if current_logps.ndim != 1 or not current_logps.numel():
        raise ValueError("trajectory log-probabilities must be a non-empty vector")
    if current_logps.shape != old_logps.shape or current_logps.shape != reference_logps.shape:
        raise ValueError("current, old and reference log-probabilities must match")
    if group_size < 2 or total_action_tokens < current_logps.numel():
        raise ValueError("invalid group/token normalization")

    advantage = torch.as_tensor(
        advantage, device=current_logps.device, dtype=torch.float32
    )
    log_ratio = (current_logps.float() - old_logps.float()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_contribution = -surrogate.mean() / group_size

    per_token_kl = positive_kl_estimate(
        current_logps.float(), reference_logps.float()
    )
    kl_contribution = float(beta) * per_token_kl.sum() / total_action_tokens
    clipped = (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
    return TrajectoryGRPOOutput(
        loss=policy_contribution + kl_contribution,
        policy_contribution=policy_contribution,
        kl_contribution=kl_contribution,
        clip_fraction=clipped.float().mean(),
        ratio_mean=ratio.mean(),
        ratio_std=ratio.std(unbiased=False),
        approx_kl=per_token_kl.mean(),
        token_count=current_logps.numel(),
    )
