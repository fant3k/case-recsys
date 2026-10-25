import unittest

import numpy as np

from recsys.pairacc_recheck import pair_accuracy_distinct, pair_accuracy_split
from recsys.review_checks import within_user_auc


class ReviewChecks(unittest.TestCase):
    def test_within_user_auc(self):
        scores = np.array([0.9, 0.1, 0.5, 0.5, 0.3, 0.7, 0.2])
        labels = np.array([True, False, True, False, False, False, False])
        users = np.array([1, 1, 2, 2, 3, 3, 3])
        r = within_user_auc(scores, labels, users)
        self.assertEqual(r["users"], 2)  # user 3 has no positive
        self.assertAlmostEqual(r["macro_auc"], (1.0 + 0.5) / 2)

    def test_pair_accuracy_distinct_drops_same_track_pairs(self):
        pred = np.array([0.5, 0.5, 0.9, 0.1]); labels = np.array([1.0, 0.2, 0.8, 0.1], dtype=np.float32)
        users = np.array([1, 1, 1, 1]); items = np.array([7, 7, 8, 9])
        r = pair_accuracy_distinct(pred, labels, users, items)
        self.assertEqual(r["pairs"], 3)  # (a,d), (c,b), (c,d); (a,b) are two listens of track 7 and are dropped
        self.assertEqual(r["users_with_pairs"], 1)

    def test_pair_accuracy_split_by_session(self):
        pred = np.array([0.9, 0.1, 0.2, 0.8]); labels = np.array([1.0, 0.1, 0.9, 0.2], dtype=np.float32)
        users = np.array([1, 1, 1, 1]); sessions = np.array([0, 0, 1, 1])
        same = pair_accuracy_split(pred, labels, users, sessions, True)
        diff = pair_accuracy_split(pred, labels, users, sessions, False)
        self.assertEqual((same["pairs"], same["macro_pair_accuracy"]), (2, 0.5))
        self.assertEqual(diff["pairs"], 2)


if __name__ == "__main__":
    unittest.main()
