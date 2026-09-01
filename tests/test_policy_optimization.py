import unittest

import torch

from trainer.policy_optimization import (
    compute_policy_loss,
    distributed_token_mean_scale,
    effective_group_mask,
    group_relative_advantages,
    positive_kl_estimate,
    soft_overlong_penalty,
)


class PolicyOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.current = torch.tensor(
            [[-1.00, -1.20, -1.40], [-0.70, -0.80, -0.90]], requires_grad=True
        )
        self.old = torch.tensor([[-1.05, -1.20, -1.35], [-0.75, -0.85, -0.95]])
        self.reference = torch.tensor([[-1.10, -1.30, -1.50], [-0.80, -0.90, -1.00]])
        self.mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.advantages = torch.tensor([1.0, -1.0])

    def test_all_objectives_are_finite_and_differentiable(self):
        for name in ("grpo", "cispo", "dapo", "gspo"):
            current = self.current.detach().clone().requires_grad_(True)
            output = compute_policy_loss(
                loss_type=name,
                current_logps=current,
                old_logps=self.old,
                reference_logps=self.reference,
                advantages=self.advantages,
                completion_mask=self.mask,
                beta=0.01,
            )
            self.assertTrue(torch.isfinite(output.loss))
            self.assertGreaterEqual(output.approx_kl.item(), 0.0)
            output.loss.backward()
            self.assertIsNotNone(current.grad)
            self.assertTrue(torch.isfinite(current.grad).all())

        # The CISPO paper uses an upper bound of 1 + epsilon_high_IS.
        current = torch.log(torch.tensor([[5.5]])).requires_grad_(True)
        output = compute_policy_loss(
            loss_type="cispo",
            current_logps=current,
            old_logps=torch.zeros_like(current),
            advantages=torch.ones(1),
            completion_mask=torch.ones_like(current),
            cispo_epsilon_high=5.0,
        )
        output.loss.backward()
        self.assertAlmostEqual(current.grad.item(), -5.5, places=5)

    def test_zero_policy_change_has_unit_ratio_and_zero_kl(self):
        current = self.old.detach().clone().requires_grad_(True)
        output = compute_policy_loss(
            loss_type="gspo",
            current_logps=current,
            old_logps=self.old,
            reference_logps=current.detach(),
            advantages=self.advantages,
            completion_mask=self.mask,
        )
        self.assertAlmostEqual(output.ratio_mean.item(), 1.0, places=6)
        self.assertAlmostEqual(output.approx_kl.item(), 0.0, places=6)

    def test_group_advantages_and_dynamic_sampling(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        advantages = group_relative_advantages(rewards, group_size=3)
        self.assertAlmostEqual(advantages[:3].mean().item(), 0.0, places=5)
        mask = effective_group_mask(rewards.bool(), group_size=3)
        self.assertEqual(mask.tolist(), [True, False])

    def test_positive_kl_and_overlong_penalty(self):
        kl = positive_kl_estimate(torch.tensor([[-1.0]]), torch.tensor([[-2.0]]))
        self.assertGreaterEqual(kl.item(), 0.0)
        extreme_kl = positive_kl_estimate(
            torch.tensor([[-1000.0]]), torch.tensor([[0.0]])
        )
        self.assertTrue(torch.isfinite(extreme_kl).all())
        zero_beta = compute_policy_loss(
            loss_type="grpo",
            current_logps=torch.tensor([[-1000.0]], requires_grad=True),
            old_logps=torch.tensor([[-1000.0]]),
            reference_logps=torch.tensor([[0.0]]),
            advantages=torch.ones(1),
            completion_mask=torch.ones(1, 1),
            beta=0.0,
        )
        self.assertTrue(torch.isfinite(zero_beta.loss))
        penalty = soft_overlong_penalty(torch.tensor([70, 80, 90, 100, 110]), 100, 20)
        self.assertEqual(penalty.tolist(), [0.0, -0.0, -0.5, -1.0, -1.0])
        self.assertEqual(distributed_token_mean_scale(self.mask).item(), 1.0)

    def test_dapo_clip_higher_differs_from_symmetric_grpo(self):
        current = torch.log(torch.tensor([[1.25]]))
        old = torch.zeros_like(current)
        kwargs = dict(
            current_logps=current,
            old_logps=old,
            advantages=torch.ones(1),
            completion_mask=torch.ones_like(current),
        )
        grpo = compute_policy_loss(loss_type="grpo", grpo_epsilon=0.2, **kwargs)
        dapo = compute_policy_loss(
            loss_type="dapo", dapo_epsilon_low=0.2,
            dapo_epsilon_high=0.28, **kwargs,
        )
        self.assertAlmostEqual(grpo.policy_loss.item(), -1.2, places=5)
        self.assertAlmostEqual(dapo.policy_loss.item(), -1.25, places=5)
        self.assertEqual(grpo.clip_fraction.item(), 1.0)
        self.assertEqual(dapo.clip_fraction.item(), 0.0)

    def test_token_level_and_sequence_first_reductions_are_distinct(self):
        current = torch.zeros(2, 3)
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        advantages = torch.tensor([1.0, 3.0])
        kwargs = dict(
            current_logps=current,
            old_logps=current,
            advantages=advantages,
            completion_mask=mask,
        )
        grpo = compute_policy_loss(loss_type="grpo", **kwargs)
        dapo = compute_policy_loss(loss_type="dapo", **kwargs)
        self.assertAlmostEqual(grpo.policy_loss.item(), -2.0, places=6)
        self.assertAlmostEqual(dapo.policy_loss.item(), -2.5, places=6)

    def test_gspo_uses_length_normalized_sequence_ratio(self):
        current = torch.log(torch.tensor([[4.0, 1.0]]))
        output = compute_policy_loss(
            loss_type="gspo",
            current_logps=current,
            old_logps=torch.zeros_like(current),
            advantages=torch.ones(1),
            completion_mask=torch.ones_like(current),
            gspo_epsilon_low=1.0,
            gspo_epsilon_high=1.0,
        )
        # exp(mean(log(4), log(1))) is the geometric mean, 2.
        self.assertAlmostEqual(output.ratio_mean.item(), 2.0, places=6)
        self.assertAlmostEqual(output.policy_loss.item(), -2.0, places=6)


if __name__ == "__main__":
    unittest.main()
