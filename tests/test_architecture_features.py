import unittest

import torch

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
    RMSNorm,
    precompute_freqs_cis,
)


class ArchitectureFeatureTests(unittest.TestCase):
    def test_rmsnorm_is_finite_and_unit_rms_before_scale(self):
        torch.manual_seed(1)
        layer = RMSNorm(16, eps=1e-6)
        output = layer(torch.randn(4, 7, 16))
        rms = output.float().pow(2).mean(-1).sqrt()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=2e-4, rtol=2e-4))

    def test_yarn_changes_long_position_frequencies_without_nan(self):
        plain_cos, plain_sin = precompute_freqs_cis(8, end=128, rope_base=10000)
        yarn_cos, yarn_sin = precompute_freqs_cis(
            8, end=128, rope_base=10000,
            rope_scaling={
                "original_max_position_embeddings": 32,
                "factor": 4,
                "beta_fast": 32,
                "beta_slow": 1,
                "attention_factor": 1.0,
            },
        )
        self.assertTrue(torch.isfinite(yarn_cos).all() and torch.isfinite(yarn_sin).all())
        self.assertFalse(torch.allclose(plain_cos[96], yarn_cos[96]))
        self.assertFalse(torch.allclose(plain_sin[96], yarn_sin[96]))

    def test_gqa_cache_keeps_kv_heads_and_cached_forward_extends_it(self):
        config = MiniMindConfig(
            hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, vocab_size=64,
        )
        model = MiniMindForCausalLM(config).eval()
        first = model(torch.tensor([[3, 4, 5, 6]]), use_cache=True)
        key, value = first.past_key_values[0]
        self.assertEqual(tuple(key.shape), (1, 4, 2, 8))
        self.assertEqual(tuple(value.shape), (1, 4, 2, 8))
        second = model(torch.tensor([[7]]), past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(second.past_key_values[0][0].shape[1], 5)
        self.assertTrue(model.model.layers[0].self_attn.flash)

    def test_sparse_moe_has_finite_auxiliary_loss_and_router_gradients(self):
        torch.manual_seed(2)
        config = MiniMindConfig(
            hidden_size=32, num_hidden_layers=2, use_moe=True,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, vocab_size=64,
            num_experts=4, num_experts_per_tok=1,
        )
        model = MiniMindForCausalLM(config).train()
        tokens = torch.randint(3, config.vocab_size, (2, 12))
        result = model(tokens, labels=tokens)
        (result.loss + result.aux_loss).backward()
        router_grads = [
            parameter.grad for name, parameter in model.named_parameters()
            if ".mlp.gate." in name
        ]
        self.assertGreater(float(result.aux_loss), 0.0)
        self.assertTrue(router_grads)
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in router_grads))


if __name__ == "__main__":
    unittest.main()
