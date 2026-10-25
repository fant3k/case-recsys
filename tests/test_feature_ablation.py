import unittest
import numpy as np
from recsys.feature_ablation import explicit_features


class ExplicitFeatureChecks(unittest.TestCase):
    def test_frequency_recency_and_oov_positions(self):
        f = explicit_features([[3, 1, 3, 4, 0]], [[1, 1, 1, 1, 0]],
                              [[1, 2, 86401, 86402]], [172802], 3)
        np.testing.assert_allclose(f[0, :, 0], [np.log(3), np.log(2), 0])
        self.assertAlmostEqual(float(f[0, 0, 1]), 2**(-86401/86400), places=6)
        self.assertAlmostEqual(float(f[0, 1, 1]), .5, places=6)
        self.assertEqual(float(f[0, 2].sum()), 0)

    def test_target_tie_and_timestamp_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            explicit_features([[3, 4]], [[1, 1]], [[1, 2]], [2], 2)
        with self.assertRaises(ValueError):
            explicit_features([[3, 4]], [[1, 1]], [[1]], [3], 2)
        f = explicit_features([[2, 0]], [[1, 0]], [[]], [3], 2)
        self.assertEqual(float(f.sum()), 0)


if __name__ == '__main__':
    unittest.main()
