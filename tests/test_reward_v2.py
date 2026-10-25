import unittest
import numpy as np
import torch
from recsys.data import Sequences
from recsys.full_sft import prepare_feedback
from recsys.models import FeedbackPolicy
from recsys.reward_v2 import RewardModelV2, all_candidate_features, candidate_features, pair_accuracy, side_inputs


def tiny():
    rows = [{'uid': 1, 'timestamp': [1, 1, 2, 4, 5, 7, 9], 'item_id': [10, 20, 10, 30, 999, 10, 40],
             'played_ratio_pct': [100, 20, 60, 0, 90, 80, 50], 'is_organic': [1, 0, 1, 0, 0, 1, 0]}]
    data = Sequences(rows, [10, 20, 30, 40], 100, 6)
    for u in data.users:
        prepare_feedback(u)
    return data


class RewardV2Checks(unittest.TestCase):
    def test_side_inputs_align_with_context(self):
        data = tiny(); u = data.users[0]
        rb, gb, cr, co = side_inputs([(u, 1, 5)], 6)
        np.testing.assert_allclose(cr[0, :4], u['ratios'][1:5])
        np.testing.assert_array_equal(co[0, :4], u['organic'][1:5])
        self.assertTrue((cr[0, 4:] == 0).all() and (rb[0, 4:] == 0).all())

    def test_features_count_recency_and_user_item_mean(self):
        rows = torch.tensor([[3, 4, 3, 5, 1, 0]])          # items 0,1,0,2,UNK, pad
        ratios = torch.tensor([[1., .2, .6, 0., .9, 0.]])
        organic = torch.tensor([[1, 0, 1, 0, 0, 0]])
        item = torch.tensor([[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
        f = candidate_features(rows, ratios, organic, torch.tensor([0]), item)[0]
        self.assertAlmostEqual(f[0].item(), np.log1p(2), places=5)       # count
        self.assertAlmostEqual(f[1].item(), (5 - 1 - 2) / 6, places=5)    # positions from end of filled context
        self.assertAlmostEqual(f[3].item(), .8, places=5)                 # user-item mean ratio
        self.assertAlmostEqual(f[6].item(), (1 + .2 + .6 + 0) / 4, places=5)  # user mean over known only
        unseen = candidate_features(rows, ratios, organic, torch.tensor([3]), item)[0]
        self.assertEqual(unseen[2].item(), 0); self.assertEqual(unseen[1].item(), 1.5)

    def test_all_candidate_features_match_single(self):
        torch.manual_seed(0)
        rows = torch.tensor([[3, 4, 3, 5, 1, 0], [6, 6, 4, 0, 0, 0]])
        ratios = torch.rand(2, 6); organic = torch.randint(0, 2, (2, 6))
        item = torch.rand(4, 2)
        full = all_candidate_features(rows, ratios, organic, item)
        for a in range(4):
            single = candidate_features(rows, ratios, organic, torch.tensor([a, a]), item)
            torch.testing.assert_close(full[:, a], single)

    def test_all_scores_equal_forward_path(self):
        torch.manual_seed(1)
        backbone = FeedbackPolicy(4, context_size=6, width=16, layers=1, heads=2, pps=True, feedback=True)
        rm = RewardModelV2(backbone, np.random.rand(4, 2).astype(np.float32), hidden=8).eval()
        x = torch.tensor([[3, 4, 3, 5, 1, 0], [6, 6, 4, 0, 0, 0]]); mask = (x > 0).long()
        rb = torch.where(x > 0, 2, 0); gb = torch.where(x > 0, 3, 0)
        cr = torch.rand(2, 6) * mask; co = torch.randint(0, 2, (2, 6)) * mask
        allsc = rm.all_scores(x, mask, rb, gb, cr, co, organic_flag=0.0)
        for a in range(4):
            single = rm(x, mask, rb, gb, cr, co, torch.tensor([a, a]), torch.zeros(2))
            torch.testing.assert_close(allsc[:, a], single, atol=1e-5, rtol=1e-5)

    def test_pair_accuracy(self):
        r = pair_accuracy(np.array([.9, .1, .5, .4]), np.array([1., 0., .8, .2]), np.array([1, 1, 2, 2]))
        self.assertEqual(r['pairs'], 2); self.assertEqual(r['macro_pair_accuracy'], 1.0)
        r = pair_accuracy(np.array([.1, .9]), np.array([1., 0.]), np.array([1, 1]))
        self.assertEqual(r['pooled_pair_accuracy'], 0.0)


if __name__ == '__main__':
    unittest.main()
