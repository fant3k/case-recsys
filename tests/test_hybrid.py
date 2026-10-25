import unittest
import numpy as np
from recsys.hybrid import history_distribution, probability_topk


class HybridChecks(unittest.TestCase):
    def test_history_probability_keeps_multiplicity_and_excludes_specials(self):
        p = history_distribution([0, 1, 2, 3, 3, 4], 3, np.array([.2, .3, .5]))
        np.testing.assert_allclose(p, [2/3, 1/3, 0])
        fallback = history_distribution([1, 2], 3, np.array([.2, .3, .5]))
        np.testing.assert_allclose(fallback, [.2, .3, .5])

    def test_normalized_mixture_endpoints_and_deterministic_ties(self):
        a = history_distribution([3, 3, 4], 3, np.ones(3)/3)
        b = history_distribution([4, 5, 5], 3, np.ones(3)/3)
        for weight in [0, .25, .5, 1]:
            mixed = (1-weight)*a+weight*b
            self.assertAlmostEqual(float(mixed.sum()), 1., places=6)
            self.assertTrue(np.all(mixed >= 0))
        self.assertEqual(probability_topk(np.array([.5, .5, 0]), np.array([20, 10, 30]), 2), [10, 20])


if __name__ == '__main__':
    unittest.main()
