import unittest

import numpy as np

from recsys.multiplicity import interval, paired_ndcg


class Multiplicity(unittest.TestCase):
    def test_interval_brackets_the_mean_and_widens_with_level(self):
        values = np.random.default_rng(0).normal(0.01, 0.05, 300)
        narrow = interval(values, 0.95, 1, draws=2000)
        wide = interval(values, 1 - 0.05 / 6, 1, draws=2000)
        self.assertLess(narrow["interval"][0], narrow["mean"])
        self.assertGreater(narrow["interval"][1], narrow["mean"])
        self.assertLess(wide["interval"][0], narrow["interval"][0])
        self.assertGreater(wide["interval"][1], narrow["interval"][1])

    def test_same_seed_is_reproducible(self):
        values = [0.1, -0.2, 0.3, 0.0, 0.05] * 20
        self.assertEqual(interval(values, 0.975, 7, draws=500), interval(values, 0.975, 7, draws=500))

    def test_paired_difference_scales_the_reference_and_checks_users(self):
        a = [{"uid": 1, "ndcg": 0.5}, {"uid": 2, "ndcg": 0.2}]
        b = [{"uid": 1, "ndcg": 0.4}, {"uid": 2, "ndcg": 0.4}]
        self.assertEqual([round(x, 6) for x in paired_ndcg(a, b)], [0.1, -0.2])
        self.assertEqual([round(x, 6) for x in paired_ndcg(a, b, 0.95)], [0.12, -0.18])
        with self.assertRaises(ValueError):
            paired_ndcg(a, [{"uid": 1, "ndcg": 0.4}, {"uid": 3, "ndcg": 0.4}])


if __name__ == "__main__":
    unittest.main()
