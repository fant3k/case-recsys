import unittest
import numpy as np
from recsys.data import Sequences
from recsys.targets import continuation_labels, length_bucket, quantile_labels


def user(times, ratios, items=None):
    items = items or [10] * len(times)
    return Sequences([{'uid': 1, 'timestamp': times, 'item_id': items, 'played_ratio_pct': ratios}], [10, 20], 10**9, 8).users[0]


class TargetChecks(unittest.TestCase):
    def test_continuation_counts_later_completed_listens_in_session(self):
        # session A: t=0,100,100,200 ; gap >1200 ; session B: t=5000,5100
        u = user([0, 100, 100, 200, 5000, 5100], [90, 90, 30, 80, 90, 90])
        lab = continuation_labels(u, cap=10**9) * 10
        # t=0 -> later in session with >50%: t=100 (90), t=200 (80) -> 2 ; tie at t=100 not counted for its mates
        np.testing.assert_allclose(lab, [2, 1, 1, 0, 1, 0])

    def test_continuation_censored_at_cap_and_capped_at_ten(self):
        u = user([0, 10, 20, 30], [90, 90, 90, 90])
        np.testing.assert_allclose(continuation_labels(u, cap=25) * 10, [2, 1, 0, 0])
        many = user(list(range(0, 1500, 100)), [90] * 15)
        self.assertEqual(continuation_labels(many, cap=10**9)[0], 1.0)

    def test_quantile_uses_only_strictly_earlier_listens_in_bucket(self):
        lengths = np.full(8, 200.0)
        u = user([0, 1, 2, 3, 4, 5, 5, 6], [10, 20, 30, 40, 50, 60, 5, 100])
        q = quantile_labels(u, lengths, min_prior=5)
        self.assertTrue(np.all(q[:5] == 0.5))            # fewer than 5 prior listens
        self.assertAlmostEqual(q[5], 1.0)                 # 60% beats all 5 earlier
        self.assertAlmostEqual(q[6], 0.0)                 # tie-mate at t=5 does not see the 60%
        self.assertAlmostEqual(q[7], 1.0)
        later = quantile_labels(user([0, 1, 2, 3, 4, 5, 5, 6, 7], [10, 20, 30, 40, 50, 60, 5, 100, 0]),
                                np.full(9, 200.0), min_prior=5)
        np.testing.assert_allclose(later[:8], q)          # appending the future changes nothing before it

    def test_length_buckets_separate_short_and_long(self):
        self.assertLess(length_bucket(np.array([60.0]))[0], length_bucket(np.array([400.0]))[0])


if __name__ == '__main__':
    unittest.main()
