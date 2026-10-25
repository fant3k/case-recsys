import math
import unittest

import numpy as np
import polars as pl
import torch

from recsys.data import Sequences, build_catalog, encode_items, pack_contexts
from recsys.metrics import ranking_metrics
from recsys.models import Policy, exact_kl, policy_loss


class DataChecks(unittest.TestCase):
    def test_tied_targets_never_enter_their_history(self):
        row = {"uid": 7, "timestamp": [10, 10, 20, 20], "item_id": [2, 1, 4, 3], "played_ratio_pct": [90]*4}
        dataset = Sequences([row], [1, 2, 3, 4], train_end=21, context_size=8)
        x, mask, actions, _, records = dataset.sample(np.random.default_rng(42), 40)
        self.assertTrue(np.all(mask.sum(1) == 2))
        self.assertTrue(set(actions).issubset({2, 3}))
        self.assertTrue(all(t == 20 for _, _, t in records))
        self.assertTrue(np.all(x[:, :2] == [3, 4]))

    def test_unknown_history_is_not_deleted(self):
        self.assertEqual(encode_items(np.array([10, 999, 20]), np.array([10, 20])).tolist(), [3, 1, 4])
        row = {"uid": 1, "timestamp": [1, 2, 3], "item_id": [10, 999, 20], "played_ratio_pct": [0, 50, 100]}
        dataset = Sequences([row], [10, 20], 4, context_size=8)
        x, mask, _, _, _ = dataset.sample(np.random.default_rng(1), 4)
        self.assertTrue(np.all(x[:, :2] == [3, 1]))
        self.assertTrue(np.all(mask.sum(1) == 2))

    def test_future_and_boundary_do_not_change_catalog(self):
        past = pl.DataFrame({"timestamp": [1, 2, 3], "item_id": [10, 10, 20]})
        future = pl.DataFrame({"timestamp": [5]+[6]*100, "item_id": [999]*101})
        a = build_catalog(past, 5, 2)
        b = build_catalog(pl.concat([past, future]), 5, 2)
        np.testing.assert_array_equal(a, b)

    def test_heldout_sampling_uses_only_requested_target_window(self):
        row = {"uid": 1, "timestamp": [1, 2, 3, 4, 5], "item_id": [10]*5, "played_ratio_pct": [0]*5}
        data = Sequences([row], [10], train_end=3)
        batch = data.sample(np.random.default_rng(1), 20, bounds=(3, 5))
        self.assertTrue(all(3 <= t < 5 for _, _, t in batch[-1]))


class MetricChecks(unittest.TestCase):
    def test_oov_target_stays_in_denominator(self):
        result, _ = ranking_metrics([1], [[10, 20]], {1: {10, 999}})
        self.assertEqual(result["Recall@10"], 0.5)
        self.assertAlmostEqual(result["NDCG@10"], 1/(1+1/math.log2(3)))

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1], [[10, 10]], {1: {10}})


class ModelChecks(unittest.TestCase):
    def test_policy_can_learn_history_dependent_toy_targets(self):
        torch.manual_seed(7)
        model = Policy(4, context_size=4, width=16, layers=1, heads=2)
        x = torch.tensor([[3, 4], [4, 3], [3, 3], [4, 4]])
        mask = torch.ones_like(x)
        target = torch.tensor([0, 1, 0, 1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        for _ in range(200):
            loss = torch.nn.functional.cross_entropy(model(x, mask), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(x, mask)
            self.assertTrue(torch.equal(logits.argmax(-1), target))
            self.assertLess(torch.nn.functional.cross_entropy(logits, target).item(), 0.05)

    def test_right_padding_does_not_change_last_position_logits(self):
        torch.manual_seed(1)
        model = Policy(8, context_size=8, width=16, layers=1, heads=2).eval()
        with torch.no_grad():
            short = model(torch.tensor([[3, 4]]), torch.tensor([[1, 1]]))
            padded = model(torch.tensor([[3, 4, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))
        self.assertEqual(tuple(padded.shape), (1, 8))
        torch.testing.assert_close(short, padded, atol=1e-5, rtol=1e-5)

    def test_reinforce_enumerated_gradient_matches_exact_objective(self):
        logits = torch.tensor([[0.2, -0.3, 0.8]], requires_grad=True)
        reference = torch.tensor([[0.1, 0.4, -0.2]])
        rewards = torch.tensor([0.0, 0.5, 1.0])
        p = logits.softmax(-1)
        direct = -(p*rewards).sum() + 0.3*exact_kl(logits, reference).sum()
        expected = torch.autograd.grad(direct, logits, retain_graph=True)[0]
        terms = []
        for action in range(3):
            loss, _ = policy_loss(logits, reference, torch.tensor([action]), rewards[action:action+1], 0.25, 0.3)
            terms.append(p[0, action].detach()*loss)
        actual = torch.autograd.grad(sum(terms), logits)[0]
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_three_action_optimum_and_kl_direction(self):
        q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        r = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        beta = 0.3
        oracle = torch.softmax(q.log()+r/beta, dim=-1)
        logits = q.log().clone().requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=0.08)
        for _ in range(200):
            p = logits.softmax(-1)
            objective = (p*r).sum() - beta*(p*(p.log()-q.log())).sum()
            optimizer.zero_grad()
            (-objective).backward()
            optimizer.step()
        torch.testing.assert_close(logits.softmax(-1), oracle, atol=2e-4, rtol=2e-4)
        self.assertGreater(float((oracle*r).sum()), float((q*r).sum()))


if __name__ == "__main__":
    unittest.main()
