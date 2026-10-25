import unittest
import numpy as np
import torch
from torch.nn import functional as F
from recsys.models import FeedbackPolicy
from recsys.reward_v2 import RewardModelV2
from recsys.rl_distill import divergence, target_logp, topk_rm_scores, truncated_rewards
from recsys.rl_v2 import kl_rl_objective


class DistillChecks(unittest.TestCase):
    def test_truncated_rewards_keep_topk_and_neutral_tail(self):
        ref = F.log_softmax(torch.tensor([[2., 1., 0., -1.]]), -1)
        idx = torch.tensor([[0, 1]]); r = torch.tensor([[0.9, 0.1]])
        full = truncated_rewards(idx, r, ref)
        w = ref.exp()[0, :2]
        neutral = float((w * r[0]).sum() / w.sum())
        torch.testing.assert_close(full, torch.tensor([[0.9, 0.1, neutral, neutral]]))
        # tail keeps its relative mass under the tilted target
        tgt = target_logp(ref, full, 0.3).exp()[0]
        self.assertAlmostEqual(float(tgt[2] / tgt[3]), float(ref.exp()[0, 2] / ref.exp()[0, 3]), places=5)

    def test_reverse_kl_gradient_matches_negative_objective_over_beta(self):
        torch.manual_seed(0)
        ref = F.log_softmax(torch.randn(3, 6), -1); r = torch.rand(3, 6); beta = 0.3
        logits = torch.randn(3, 6, requires_grad=True)
        g1, = torch.autograd.grad(divergence("reverse", logits, target_logp(ref, r, beta)).sum(), logits)
        g2, = torch.autograd.grad((-kl_rl_objective(logits, ref, r, beta)[0] / beta).sum(), logits)
        torch.testing.assert_close(g1, g2, atol=1e-5, rtol=1e-4)

    def test_forward_kl_gradient_is_pi_minus_target_and_all_reach_optimum(self):
        torch.manual_seed(1)
        tgt = F.log_softmax(torch.randn(2, 5), -1)
        logits = torch.randn(2, 5, requires_grad=True)
        g, = torch.autograd.grad(divergence("forward", logits, tgt).sum(), logits)
        torch.testing.assert_close(g, torch.softmax(logits.detach(), -1) - tgt.exp())
        for kind in ("reverse", "forward", "jeffreys"):
            z = torch.zeros(2, 5, requires_grad=True); opt = torch.optim.Adam([z], lr=0.05)
            for _ in range(2000):
                opt.zero_grad(); divergence(kind, z, tgt).sum().backward(); opt.step()
            torch.testing.assert_close(F.log_softmax(z.detach(), -1), tgt, atol=3e-3, rtol=0)

    def test_topk_scores_equal_full_catalog_scores(self):
        torch.manual_seed(2)
        backbone = FeedbackPolicy(6, context_size=6, width=16, layers=1, heads=2, pps=True, feedback=True)
        rm = RewardModelV2(backbone, np.random.rand(6, 2).astype(np.float32), hidden=8).eval()
        x = torch.tensor([[3, 4, 3, 5, 1, 0], [6, 6, 4, 8, 0, 0]]); mask = (x > 0).long()
        rb = torch.where(x > 0, 2, 0); gb = torch.where(x > 0, 3, 0)
        cr = torch.rand(2, 6) * mask; co = torch.randint(0, 2, (2, 6)) * mask
        full = rm.all_scores(x, mask, rb, gb, cr, co, organic_flag=0.0)
        idx = torch.tensor([[5, 0, 2], [1, 3, 4]])
        torch.testing.assert_close(topk_rm_scores(rm, x, mask, rb, gb, cr, co, idx), full.gather(1, idx), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
