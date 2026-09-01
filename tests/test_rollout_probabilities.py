import types
import unittest

import torch

from trainer.rollout_engine import compute_per_token_logps, masked_log_softmax


class TinyLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        batch, length = input_ids.shape
        logits = self.fixed_logits[:length].unsqueeze(0).expand(batch, -1, -1)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=logits)


class RolloutProbabilityTest(unittest.TestCase):
    def test_pad_is_excluded_before_fp32_log_softmax(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
        actual = masked_log_softmax(logits, blocked_token_ids=[0])
        expected = torch.log_softmax(logits.float().masked_fill(torch.tensor([[True, False, False, False]]), -torch.inf), dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0))
        self.assertTrue(torch.isneginf(actual[0, 0]))

    def test_unupdated_policy_has_unit_ratio_on_same_legal_set(self):
        logits = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.4, 0.3, 0.2],
            [0.9, 0.1, 0.2, 0.3],
        ])
        model = TinyLM(logits)
        input_ids = torch.tensor([[1, 2, 3]])
        old = compute_per_token_logps(model, input_ids, 2, blocked_token_ids=[0])
        current = compute_per_token_logps(model, input_ids, 2, blocked_token_ids=[0])
        ratio = (current - old).exp()
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
