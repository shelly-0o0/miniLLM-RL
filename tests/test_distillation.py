import unittest

import torch

from trainer.train_distillation import distillation_loss


class DistillationLossTest(unittest.TestCase):
    def test_identical_logits_have_zero_kl(self):
        logits = torch.randn(7, 13)
        loss = distillation_loss(logits, logits, temperature=1.5)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_student_receives_finite_gradient(self):
        student = torch.randn(5, 11, requires_grad=True)
        teacher = torch.randn(5, 11)
        loss = distillation_loss(student, teacher, temperature=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
