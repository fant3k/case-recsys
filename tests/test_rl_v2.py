import unittest
import numpy as np
import torch
from torch.nn import functional as F
from recsys.data import Sequences
from recsys.full_sft import chunked_pps_cross_entropy, group_starts_per_event, sample_targets
from recsys.rl_v2 import (analytic_logp, awr_weights, kl_rl_objective, margin_interval, prefix_baseline,
                          weighted_pps_cross_entropy)


class ExactObjectiveChecks(unittest.TestCase):
    def test_expected_reward_gradient_is_closed_form(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 7, requires_grad=True)
        r = torch.rand(3, 7)
        _, expected, _ = kl_rl_objective(logits, F.log_softmax(torch.zeros(3, 7), -1), r, 0.0)
        expected.sum().backward()
        p = torch.softmax(logits.detach(), -1)
        torch.testing.assert_close(logits.grad, p * (r - (p * r).sum(-1, keepdim=True)))

    def test_kl_zero_at_reference_and_free_optimum_is_analytic(self):
        torch.manual_seed(1)
        ref = F.log_softmax(torch.randn(2, 6), -1)
        r = torch.rand(2, 6)
        _, _, kl = kl_rl_objective(ref.clone(), ref, r, 0.3)
        torch.testing.assert_close(kl, torch.zeros(2), atol=1e-6, rtol=0)
        logits = ref.clone().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=0.05)
        for _ in range(3000):
            j, _, _ = kl_rl_objective(logits, ref, r, 0.3)
            opt.zero_grad(); (-j.sum()).backward(); opt.step()
        star = analytic_logp(ref, r, 0.3)
        torch.testing.assert_close(F.log_softmax(logits.detach(), -1), star, atol=2e-3, rtol=0)
        j_star = kl_rl_objective(star, ref, r, 0.3)[0]
        j_ref = kl_rl_objective(ref, ref, r, 0.3)[0]
        self.assertTrue(torch.all(j_star >= j_ref))

    def test_analytic_limits(self):
        ref = F.log_softmax(torch.tensor([[0., 1., 2.]]), -1)
        r = torch.tensor([[1., 0., 0.]])
        torch.testing.assert_close(analytic_logp(ref, r, 1e6), ref, atol=1e-5, rtol=0)
        self.assertEqual(int(analytic_logp(ref, r, 1e-3).argmax()), 0)


class OfflineChecks(unittest.TestCase):
    def test_awr_weights(self):
        w = awr_weights([1.0, 0.5, 0.0, 0.9], [0.5, 0.5, 0.5, 0.5], 0.1)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        raw = np.minimum(np.exp((np.array([1.0, 0.5, 0.0, 0.9]) - 0.5) / 0.1), 20)
        np.testing.assert_allclose(w, raw / raw.mean(), rtol=1e-5)
        self.assertTrue(w[0] == w[3])  # both clipped at 20 before normalisation
        np.testing.assert_allclose(awr_weights([.3, .7], [.1, .1], .5), awr_weights([.5, .9], [.3, .3], .5), rtol=1e-6)

    def test_prefix_baseline_uses_only_visible_context(self):
        rows = [{'uid': 1, 'timestamp': [1, 1, 2, 4, 5, 7, 9, 9], 'item_id': [10, 20, 10, 30, 10, 40, 20, 30],
                 'played_ratio_pct': [100, 20, 60, 0, 90, 80, 50, 10]}]
        data = Sequences(rows, [10, 20, 30, 40], 100, 5)
        u = data.users[0]
        u['group_start'] = group_starts_per_event(u)
        u['ratios_cumsum'] = np.cumsum(u['ratios'], dtype=np.float64)
        for seed in range(20):
            audit = []
            sample_targets(data, np.random.default_rng(seed), 3, True, audit=audit)
            for user, start, pos, event in audit:
                self.assertGreater(event, start + pos)
                self.assertAlmostEqual(prefix_baseline(user, start, pos, 'ratios'),
                                       float(user['ratios'][start:start + pos + 1].mean()), places=6)

    def test_unit_weights_match_unweighted_loss(self):
        torch.manual_seed(2)
        h = torch.randn(9, 8); w = torch.randn(5, 8); a = torch.randint(0, 5, (9,))
        rows = torch.randint(0, 8, (9, 6)); pos = torch.randint(0, 6, (9,))
        torch.testing.assert_close(weighted_pps_cross_entropy(h, w, a, rows, pos, torch.ones(9), .01, chunk=4),
                                   chunked_pps_cross_entropy(h, w, a, rows, pos, .01, chunk=4))

    def test_margin_interval(self):
        ref = [{'uid': i, 'ndcg': .2} for i in range(50)]
        same = [{'uid': i, 'ndcg': .2} for i in range(50)]
        worse = [{'uid': i, 'ndcg': .1} for i in range(50)]
        self.assertTrue(margin_interval(same, ref)['within_5pct_supported'])
        self.assertFalse(margin_interval(worse, ref)['within_5pct_supported'])


if __name__ == '__main__':
    unittest.main()
