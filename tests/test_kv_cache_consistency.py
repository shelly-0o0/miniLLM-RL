import unittest

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class KVCacheConsistencyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = MiniMindConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=128,
            max_position_embeddings=128,
            dropout=0.0,
            flash_attn=True,
        )
        self.model = MiniMindForCausalLM(config).eval()
        self.input_ids = torch.randint(3, config.vocab_size, (2, 17))

    def test_incremental_cached_logits_match_full_forward(self):
        with torch.inference_mode():
            full_logits = self.model(
                self.input_ids,
                attention_mask=torch.ones_like(self.input_ids),
                use_cache=False,
            ).logits

            past_key_values = None
            incremental_logits = []
            for position in range(self.input_ids.size(1)):
                output = self.model(
                    self.input_ids[:, position:position + 1],
                    attention_mask=torch.ones(
                        self.input_ids.size(0), position + 1, dtype=torch.long
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                incremental_logits.append(output.logits[:, -1, :])

        cached_logits = torch.stack(incremental_logits, dim=1)
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)

    def test_cached_and_uncached_greedy_generation_match(self):
        prompt = self.input_ids[:1, :8]
        attention_mask = torch.ones_like(prompt)
        with torch.inference_mode():
            cached = self.model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=None,
                use_cache=True,
            )
            uncached = self.model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=None,
                use_cache=False,
            )
        self.assertTrue(torch.equal(cached, uncached))


if __name__ == "__main__":
    unittest.main()
