import unittest

import torch
from torch.nn import functional as F

from recsys.exploratory import CandidateHead, PAIRWISE, head_logp, leave_one_out_advantage, objective, regularised_return, reward_to_go
from recsys.rl_v2 import kl_rl_objective


class ExploratoryChecks(unittest.TestCase):
    def test_head_starts_at_reference_and_only_moves_top_k(self):
        torch.manual_seed(0)
        ref = F.log_softmax(torch.randn(2, 30), -1)
        idx = ref.topk(5, -1).indices
        head = CandidateHead(12, [1.0] * 12)
        feats = torch.randn(2, 5, 12)
        torch.testing.assert_close(head_logp(ref, idx, head(feats)), ref)
        corr = torch.zeros(2, 5); corr[:, 0] = 2.0
        out = head_logp(ref, idx, corr)
        tail = torch.ones_like(ref, dtype=torch.bool).scatter_(1, idx, False)
        shift = (out - ref)[tail].reshape(2, -1)
        torch.testing.assert_close(shift, shift[:, :1].expand_as(shift))

    def test_no_pairwise_head_ignores_pairwise_features(self):
        keep = [0.0 if i in PAIRWISE else 1.0 for i in range(12)]
        head = CandidateHead(12, keep)
        torch.nn.init.normal_(head.net[-1].weight)
        feats = torch.randn(3, 5, 12)
        changed = feats.clone(); changed[..., PAIRWISE] += 10
        torch.testing.assert_close(head(feats), head(changed))

    def test_objective_matches_rl_objective(self):
        torch.manual_seed(1)
        ref = F.log_softmax(torch.randn(3, 8), -1); logits = torch.randn(3, 8); r = torch.rand(3, 8)
        torch.testing.assert_close(objective(F.log_softmax(logits, -1), ref, r, 0.3), kl_rl_objective(logits, ref, r, 0.3)[0])

    def test_reward_to_go(self):
        r = torch.tensor([[1., 0.], [2., 1.], [3., 5.]])
        torch.testing.assert_close(reward_to_go(r), torch.tensor([[6., 6.], [5., 6.], [3., 5.]]))

    def test_finite_mask_keeps_kl_gradient_finite(self):
        logits = torch.randn(2, 6, requires_grad=True); ref = torch.randn(2, 6)
        idx = torch.tensor([[1], [4]])
        lp = F.log_softmax(logits.scatter(1, idx, -1e9), -1); rp = F.log_softmax(ref.scatter(1, idx, -1e9), -1)
        kl = (lp.exp() * (lp - rp)).sum(-1).sum()
        kl.backward()
        self.assertTrue(torch.isfinite(kl) and torch.isfinite(logits.grad).all())

    def test_regularised_return_credits_only_later_penalties(self):
        r = torch.tensor([[1.0], [2.0], [3.0]]); pen = torch.tensor([[0.1], [0.2], [0.4]])
        torch.testing.assert_close(regularised_return(r, pen), torch.tensor([[6.0 - 0.6], [5.0 - 0.4], [3.0]]))

    def test_leave_one_out_advantage(self):
        r = torch.tensor([[1.0, 2.0, 6.0]])
        torch.testing.assert_close(leave_one_out_advantage(r), torch.tensor([[1.0 - 4.0, 2.0 - 3.5, 6.0 - 1.5]]))


if __name__ == "__main__":
    unittest.main()
