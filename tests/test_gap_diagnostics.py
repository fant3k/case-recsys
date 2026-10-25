import unittest
import numpy as np
from recsys.gap_diagnostics import history_indices, rank_within_history


class GapDiagnosticChecks(unittest.TestCase):
    def test_history_indices_drop_special_tokens(self):
        self.assertEqual(history_indices([0, 1, 2, 3, 5, 3]).tolist(), [0, 2, 0])

    def test_ranks_history_by_scores_then_fills_without_duplicates(self):
        scores = np.array([.1, .5, .2, .9, .3])
        history = np.array([0, 2, 2])
        fill = np.array([.0, .4, .0, .8, .1])
        self.assertEqual(rank_within_history(scores, history, fill, k=4), [2, 0, 3, 1])
        counts = np.bincount(history, minlength=5).astype(float)
        self.assertEqual(rank_within_history(counts, history, fill, k=4), [2, 0, 3, 1])

    def test_only_history_when_enough_items(self):
        scores = np.array([.3, .1, .2, .9])
        self.assertEqual(rank_within_history(scores, np.array([0, 1, 2]), scores, k=2), [0, 2])


if __name__ == '__main__':
    unittest.main()
