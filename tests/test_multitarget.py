import unittest
import numpy as np
import torch
from recsys.data import Sequences
from recsys.models import Policy
from recsys.multitarget import causal_targets, logits_at_positions


class MultipleTargetChecks(unittest.TestCase):
    def test_group_targets_start_before_entire_tie_and_omit_oov(self):
        row = {'uid': 1, 'timestamp': [1, 1, 2, 2, 3, 4, 4],
               'item_id': [10, 20, 30, 40, 999, 10, 20], 'played_ratio_pct': [90]*7}
        data = Sequences([row], [10, 20, 30, 40], 5, 8)
        # Anchor group4; input includes whole group2, but their logits must be at group1 end.
        x = np.array([[3, 4, 5, 6, 1, 0, 0, 0]])
        mask = np.array([[1, 1, 1, 1, 1, 0, 0, 0]])
        batch = (x, mask, np.array([0]), None, [(1, 5, 4)])
        owners, positions, actions, weights, lengths = causal_targets(batch, {1: data.users[0]}, 8, np.random.default_rng(1))
        self.assertEqual(positions.tolist(), [4, 1])
        self.assertEqual(actions[0], 0)
        self.assertIn(actions[1], [2, 3])
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertEqual(lengths.tolist(), [5, 2])

    def test_position_logits_match_prefix_and_ignore_future_tokens(self):
        torch.manual_seed(42)
        model = Policy(8, context_size=8, width=16, layers=1, heads=2).eval()
        x = torch.tensor([[3, 4, 5, 6]])
        mask = torch.ones_like(x)
        with torch.no_grad():
            multi = logits_at_positions(model, x, mask, torch.tensor([0, 0]), torch.tensor([1, 3]))
            short = model(x[:, :2], mask[:, :2])
            last = model(x, mask)
            changed = logits_at_positions(model, torch.tensor([[3, 4, 9, 10]]), mask, torch.tensor([0]), torch.tensor([1]))
        torch.testing.assert_close(multi[0:1], short, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(multi[1:2], last, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(changed, short, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
