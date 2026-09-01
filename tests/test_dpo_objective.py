import math
import unittest

import torch

from trainer.train_dpo import dpo_loss


class DPOObjectiveTest(unittest.TestCase):
    def test_identical_policy_and_reference_have_log_two_loss(self):
        reference = torch.tensor([[1.0], [0.0]])
        policy = reference.clone().requires_grad_(True)
        mask = torch.ones_like(reference)
        loss = dpo_loss(reference, policy, mask, beta=0.15)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(policy.grad).all())

    def test_increasing_chosen_relative_margin_lowers_loss(self):
        reference = torch.zeros(2, 1)
        mask = torch.ones_like(reference)
        neutral = dpo_loss(reference, torch.zeros(2, 1), mask, beta=1.0)
        improved = dpo_loss(
            reference,
            torch.tensor([[2.0], [-1.0]]),
            mask,
            beta=1.0,
        )
        self.assertLess(improved.item(), neutral.item())


if __name__ == "__main__":
    unittest.main()
