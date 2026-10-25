import math
import unittest
import numpy as np
from recsys.data import Sequences
from recsys.diagnostics import decayed_counts, fit_transitions, slice_metrics
from recsys.metrics import ranking_metrics


class DiagnosticChecks(unittest.TestCase):
    def test_transitions_do_not_invent_tie_order_or_bridge_oov_or_use_future(self):
        row = {'uid': 1, 'timestamp': [1, 1, 2, 3, 4, 5],
               'item_id': [10, 20, 30, 999, 10, 40], 'played_ratio_pct': [100]*6}
        data = Sequences([row], [10, 20, 30, 40], train_end=5)
        edges, _ = fit_transitions(data)
        self.assertEqual(edges[0], {2: 1.0})
        self.assertEqual(edges[1], {2: 1.0})
        self.assertEqual(edges.get(2, {}), {})
        self.assertNotIn(3, edges.get(0, {}))

    def test_slice_contributions_add_to_original_ndcg_and_keep_oov(self):
        truth = {1: {10, 20, 30, 999}}
        recs = [[10, 30, 20]]
        result, rows = slice_metrics([1], recs, truth, {1: {10, 20}}, {1: {10}}, [10, 20, 30])
        overall, _ = ranking_metrics([1], recs, truth)
        self.assertAlmostEqual(rows[0]['ndcg'], overall['NDCG@10'])
        self.assertAlmostEqual(sum(s['ndcg_contribution_all_active_users'] for s in result['slices'].values()), overall['NDCG@10'])
        self.assertEqual(result['slices']['never_seen']['target_pairs'], 2)
        self.assertEqual(result['slices']['never_seen']['catalog_target_pairs'], 1)
        self.assertEqual(result['slices']['never_seen']['pooled_recall'], .5)

    def test_time_decay_preserves_stale_history_relative_weights(self):
        scores = decayed_counts([3, 4, 4], [1, 3601, 3601], 99999999, 3600)
        self.assertAlmostEqual(scores[0], .5)
        self.assertAlmostEqual(scores[1], 2.)
        with self.assertRaises(ValueError):
            decayed_counts([3], [10], 10, 3600)


if __name__ == '__main__':
    unittest.main()
