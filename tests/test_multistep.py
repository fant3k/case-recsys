import unittest

import numpy as np
import torch

from recsys.data import OFFSET
from recsys.full_sft import prepare_feedback
from recsys.multistep import append_event, gap_bucket, listened_seconds, ndcg_at, ratio_bucket


class MultistepChecks(unittest.TestCase):
    def test_buckets_match_training_feedback(self):
        ratios = np.array([0.0, 0.1, 0.2, 0.49, 0.5, 0.79, 0.8, 1.0], dtype=np.float32)
        times = np.cumsum([0, 1, 3, 10, 100, 1000, 70000, 5]).astype(np.int64)
        user = {"ratios": ratios, "times": times}
        prepare_feedback(user)
        torch.testing.assert_close(ratio_bucket(torch.as_tensor(ratios)), torch.as_tensor(user["ratio_bucket"]))
        gaps = torch.as_tensor(np.diff(times))
        torch.testing.assert_close(gap_bucket(gaps), torch.as_tensor(user["gap_bucket"][1:]))

    def test_append_fills_padding_then_drops_oldest(self):
        x = torch.tensor([[5, 6, 0], [5, 6, 7]]); mask = (x > 0).long()
        rb = torch.tensor([[2, 3, 0], [2, 3, 4]]); gb = torch.tensor([[22, 4, 0], [22, 4, 5]])
        cr = torch.tensor([[.1, .3, 0.], [.1, .3, .6]]); co = torch.tensor([[1, 1, 0], [1, 1, 1]])
        out = append_event([x, mask, rb, gb, cr, co], torch.tensor([9, 9]), torch.tensor([1.0, 0.0]), torch.tensor([200.0, 200.0]))
        self.assertEqual(out[0].tolist(), [[5, 6, 9 + OFFSET], [6, 7, 9 + OFFSET]])
        self.assertEqual(out[1].tolist(), [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(out[2].tolist(), [[2, 3, 5], [3, 4, 1]])
        self.assertEqual(out[3][:, -1].tolist(), [8, 8])
        self.assertEqual(out[5].tolist(), [[1, 1, 0], [1, 1, 0]])
        self.assertEqual(x.tolist(), [[5, 6, 0], [5, 6, 7]])  # inputs untouched

    def test_slate_ndcg_keeps_positions_and_zeroes_repeats(self):
        self.assertAlmostEqual(ndcg_at([1, 1, 2, 3, 4], {2}), 1 / np.log2(4))  # hit stays at position 3
        self.assertAlmostEqual(ndcg_at([2, 9, 9, 9, 9], {2, 3}), 1 / (1 + 1 / np.log2(3)))

    def test_listened_seconds(self):
        torch.testing.assert_close(listened_seconds(torch.tensor([0.0, 0.5, 1.3]), torch.tensor([200.0, 200.0, 200.0])),
                                   torch.tensor([0.0, 100.0, 200.0]))


if __name__ == "__main__":
    unittest.main()
