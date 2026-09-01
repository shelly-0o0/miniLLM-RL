"""Policy-optimization objectives shared by MiniMind RL trainers.

This module keeps the math independent from rollout, reward, logging and model
code so that GRPO/CISPO/DAPO/GSPO can be unit-tested on small tensors.

Implementation provenance
-------------------------
* GRPO: DeepSeekMath, arXiv:2402.03300.
* CISPO: MiniMax-M1, arXiv:2506.13585.
* DAPO: Yu et al., arXiv:2503.14476 (Clip-Higher, Dynamic Sampling,
  token-level policy-gradient loss and soft overlong punishment).
* GSPO: Zheng et al., arXiv:2507.18071 (sequence-level importance ratio).

The surrounding training loop is adapted from MiniMind's original
``trainer/train_grpo.py`` and ``trainer/train_agent.py``.  This file is an
educational reproduction, not copied code from any third-party implementation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class PolicyLossOutput:
    """Loss and auditable diagnostics for one policy update."""

    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    ratio_mean: Tensor
    ratio_std: Tensor
    token_count: Tensor


def _zero_like(reference: Tensor) -> Tensor:
    """Return a differentiable scalar zero on ``reference``'s device."""

    return reference.sum() * 0.0


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over valid entries, returning a differentiable zero if empty."""

    mask = mask.to(values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return _zero_like(values)
    return (values * mask).sum() / denominator


def distributed_token_mean_scale(mask: Tensor) -> Tensor:
    """Gradient scale that turns DDP local token means into a global mean.

    DDP averages gradients across ranks.  A plain local token mean therefore
    gives every rank the same weight even when response lengths differ.  If a
    rank owns ``n_r`` valid tokens and the global count is ``N``, multiplying
    its local mean by ``world_size * n_r / N`` yields exactly the gradient of
    the global token sum divided by ``N`` after DDP's gradient averaging.

    The caller should apply this only to token-mean objectives (CISPO/DAPO),
    and before ``backward``.  It returns one outside distributed execution.
    """

    local_count = mask.detach().to(dtype=torch.float32).sum()
    if not dist.is_available() or not dist.is_initialized():
        return local_count.new_ones(())
    global_count = local_count.clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        return local_count.new_zeros(())
    return local_count * dist.get_world_size() / global_count


def group_relative_advantages(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    """Standardize rewards independently inside every prompt group.

    Args:
        rewards: Flat ``[num_prompts * group_size]`` reward vector.
        group_size: Number of sampled completions for each prompt.
        eps: Numerical stabilizer.  Zero-variance groups naturally produce
            zero advantages; DAPO dynamic sampling can remove them earlier.
    """

    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape={tuple(rewards.shape)}")
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("group_size must be >=2 and divide the reward count")
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - means) / (stds + eps)).reshape(-1)


def effective_group_mask(task_success: Tensor, group_size: int) -> Tensor:
    """Return DAPO's dynamic-sampling mask at prompt-group granularity.

    A group is effective iff at least one rollout succeeds and at least one
    fails.  ``task_success`` must be the *binary, verifiable* task outcome,
    not a dense shaped reward.
    """

    if task_success.ndim != 1 or task_success.numel() % group_size:
        raise ValueError("task_success must be flat and divisible by group_size")
    successes = task_success.to(torch.bool).view(-1, group_size).sum(dim=1)
    return (successes > 0) & (successes < group_size)


def expand_group_mask(group_mask: Tensor, group_size: int) -> Tensor:
    """Expand a prompt-level mask to its sampled completion rows."""

    if group_mask.ndim != 1:
        raise ValueError("group_mask must be 1-D")
    return group_mask.repeat_interleave(group_size)


def soft_overlong_penalty(lengths: Tensor, max_length: int, cache_length: int) -> Tensor:
    """DAPO soft overlong punishment in ``[-1, 0]``.

    No penalty is applied at or below ``max_length - cache_length``.  The
    penalty then decreases linearly to -1 at ``max_length`` and is clipped for
    longer responses.  Add the result to a verifiable reward before computing
    group-relative advantages.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0 < cache_length <= max_length:
        raise ValueError("cache_length must be in (0, max_length]")
    start = max_length - cache_length
    return (-((lengths.to(torch.float32) - start) / cache_length).clamp(0.0, 1.0)).to(lengths.device)


def positive_kl_estimate(current_logps: Tensor, reference_logps: Tensor) -> Tensor:
    """Non-negative k3 Monte-Carlo KL estimator used by GRPO-family code.

    With ``delta = log pi_ref - log pi_theta``, k3 is
    ``exp(delta) - delta - 1``.  It is pointwise non-negative and is more
    interpretable for monitoring than the signed sample mean of ``delta``.
    """

    # A single near-zero current probability can otherwise overflow exp in
    # FP32 and turn ``beta=0 * inf`` into NaN.  A |log-ratio| of 20 already
    # represents an extreme probability ratio (~4.85e8), so clipping remains
    # a loud diagnostic while keeping the objective finite.
    delta = (reference_logps - current_logps).clamp(min=-20.0, max=20.0)
    return torch.exp(delta) - delta - 1.0


def compute_policy_loss(
    *,
    loss_type: str,
    current_logps: Tensor,
    old_logps: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
    reference_logps: Optional[Tensor] = None,
    beta: float = 0.0,
    grpo_epsilon: float = 0.2,
    cispo_epsilon_high: float = 5.0,
    dapo_epsilon_low: float = 0.2,
    dapo_epsilon_high: float = 0.28,
    gspo_epsilon_low: float = 3e-4,
    gspo_epsilon_high: float = 4e-4,
) -> PolicyLossOutput:
    """Compute GRPO, CISPO, DAPO or GSPO on already sampled completions.

    ``current_logps``, ``old_logps`` and ``completion_mask`` have shape
    ``[N, T]``.  ``advantages`` may be trajectory-level ``[N]`` or token-level
    ``[N, T]``.  In MiniMind's current RLVR pipeline it is trajectory-level.

    Reduction differences are intentional:
    * GRPO: mean tokens per sequence, then mean sequences.
    * CISPO/DAPO: global mean over all valid response tokens (token-level loss).
    * GSPO: one importance ratio and one surrogate loss per sequence.
    """

    loss_type = loss_type.lower()
    if loss_type not in {"grpo", "cispo", "dapo", "gspo"}:
        raise ValueError(f"unsupported loss_type={loss_type!r}")
    if current_logps.shape != old_logps.shape or current_logps.shape != completion_mask.shape:
        raise ValueError("current_logps, old_logps and completion_mask must have identical shapes")
    if current_logps.ndim != 2:
        raise ValueError("log-probability tensors must have shape [N, T]")
    if advantages.ndim == 1:
        if advantages.size(0) != current_logps.size(0):
            raise ValueError("trajectory advantages must have shape [N]")
        token_advantages = advantages.unsqueeze(1).expand_as(current_logps)
        sequence_advantages = advantages
    elif advantages.shape == current_logps.shape:
        token_advantages = advantages
        valid = completion_mask.to(current_logps.dtype)
        sequence_advantages = (advantages * valid).sum(1) / valid.sum(1).clamp(min=1)
    else:
        raise ValueError("advantages must have shape [N] or [N, T]")

    mask = completion_mask.to(current_logps.dtype)
    token_counts = mask.sum(dim=1)
    valid_rows = token_counts > 0
    safe_current = current_logps.float()
    safe_old = old_logps.float()
    log_ratio = (safe_current - safe_old).clamp(min=-20.0, max=20.0)
    token_ratio = torch.exp(log_ratio)

    if reference_logps is None:
        per_token_kl = torch.zeros_like(safe_current)
    else:
        if reference_logps.shape != current_logps.shape:
            raise ValueError("reference_logps must match current_logps")
        per_token_kl = positive_kl_estimate(safe_current, reference_logps.float())
    approx_kl = masked_mean(per_token_kl.detach(), mask)
    kl_loss = masked_mean(per_token_kl, mask) * float(beta)

    if loss_type == "cispo":
        # CISPO clips the sampling correction coefficient, stops its gradient,
        # and differentiates log pi directly.  The paper defines the upper
        # bound as 1 + epsilon_high_IS; upstream MiniMind instead treated its
        # CLI value as the absolute bound, which is intentionally corrected.
        cispo_upper = 1.0 + cispo_epsilon_high
        coefficient = token_ratio.clamp(max=cispo_upper).detach()
        per_token_policy = -(coefficient * token_advantages * safe_current)
        clipped = token_ratio > cispo_upper
        ratio_for_metrics = token_ratio
    elif loss_type in {"grpo", "dapo"}:
        low, high = ((grpo_epsilon, grpo_epsilon) if loss_type == "grpo"
                     else (dapo_epsilon_low, dapo_epsilon_high))
        clipped_ratio = token_ratio.clamp(1.0 - low, 1.0 + high)
        surrogate = torch.minimum(token_ratio * token_advantages, clipped_ratio * token_advantages)
        per_token_policy = -surrogate
        clipped = (token_ratio < 1.0 - low) | (token_ratio > 1.0 + high)
        ratio_for_metrics = token_ratio
    else:
        # GSPO uses the length-normalized *sequence* likelihood ratio.  The
        # ratio is broadcast only for common diagnostics; its loss is reduced
        # once per sequence below.
        seq_log_ratio = (log_ratio * mask).sum(1) / token_counts.clamp(min=1)
        seq_ratio = torch.exp(seq_log_ratio)
        clipped_seq_ratio = seq_ratio.clamp(1.0 - gspo_epsilon_low, 1.0 + gspo_epsilon_high)
        seq_surrogate = torch.minimum(
            seq_ratio * sequence_advantages,
            clipped_seq_ratio * sequence_advantages,
        )
        policy_loss = -seq_surrogate[valid_rows].mean() if valid_rows.any() else _zero_like(current_logps)
        clipped_seq = (seq_ratio < 1.0 - gspo_epsilon_low) | (seq_ratio > 1.0 + gspo_epsilon_high)
        clip_fraction = clipped_seq[valid_rows].float().mean() if valid_rows.any() else _zero_like(current_logps)
        ratio_values = seq_ratio[valid_rows]
        ratio_mean = ratio_values.mean() if ratio_values.numel() else _zero_like(current_logps)
        ratio_std = ratio_values.std(unbiased=False) if ratio_values.numel() else _zero_like(current_logps)
        return PolicyLossOutput(
            loss=policy_loss + kl_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            token_count=mask.sum().detach(),
        )

    if loss_type in {"cispo", "dapo"}:
        # Both DAPO and the published CISPO objective use token-level
        # averaging, unlike the sequence-first MiniMind GRPO baseline.
        policy_loss = masked_mean(per_token_policy, mask)
    else:
        per_sequence = (per_token_policy * mask).sum(1) / token_counts.clamp(min=1)
        policy_loss = per_sequence[valid_rows].mean() if valid_rows.any() else _zero_like(current_logps)

    clip_fraction = masked_mean(clipped.to(mask.dtype), mask)
    valid_ratios = ratio_for_metrics[mask.bool()]
    ratio_mean = valid_ratios.mean() if valid_ratios.numel() else _zero_like(current_logps)
    ratio_std = valid_ratios.std(unbiased=False) if valid_ratios.numel() else _zero_like(current_logps)
    return PolicyLossOutput(
        loss=policy_loss + kl_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        token_count=mask.sum().detach(),
    )
