"""Numerics helpers of the precision experiment (docs/PRECISION-PLAN.md)."""
import unittest

import torch

from recsys import rl_distill, rl_v2


class GradientStats(unittest.TestCase):
    def test_identical_gradients_have_cosine_one_and_zero_error(self):
        truth = {"transformer.wte.weight": torch.tensor([[1.0, -2.0], [0.5, 0.0]]), "head.bias": torch.tensor([3.0, 4.0])}
        stats = rl_distill.gradient_stats({k: v.clone() for k, v in truth.items()}, truth)
        self.assertAlmostEqual(stats["all"]["cosine_to_fp32"], 1.0, places=12)
        self.assertEqual(stats["all"]["relative_l2_error"], 0.0)
        self.assertAlmostEqual(stats["token_embedding"]["norm_ratio"], 1.0, places=12)

    def test_flushed_entries_show_up_as_zero_fraction_and_lower_norm(self):
        truth = {"transformer.wte.weight": torch.tensor([1e-8, 1e-8, 1e-8, 1.0])}
        flushed = {"transformer.wte.weight": torch.tensor([0.0, 0.0, 0.0, 1.0])}
        stats = rl_distill.gradient_stats(flushed, truth)["token_embedding"]
        self.assertEqual(stats["zero_fraction"], 0.75)
        self.assertEqual(stats["zero_fraction_fp32"], 0.0)
        self.assertEqual(stats["fp32_fraction_below_fp16_min_subnormal"], 0.75)
        self.assertLess(stats["norm_ratio"], 1.0 + 1e-6)
        self.assertGreater(stats["cosine_to_fp32"], 0.999999)

    def test_model_without_embedding_has_no_embedding_entry(self):
        stats = rl_distill.gradient_stats({"w": torch.ones(2)}, {"w": torch.ones(2)})
        self.assertIsNone(stats["token_embedding"])


class TrainAutocast(unittest.TestCase):
    def test_fp32_is_a_plain_context_and_computes_in_float32(self):
        with rl_distill.train_autocast("fp32"):
            self.assertEqual((torch.ones(2, 2) @ torch.ones(2, 2)).dtype, torch.float32)

    def test_historical_precisions_use_the_shared_fp16_autocast(self):
        for name in ("fp16", "fp16scaled"):
            with self.subTest(precision=name):
                self.assertIs(type(rl_distill.train_autocast(name)), type(rl_v2.autocast()))


if __name__ == "__main__":
    unittest.main()
