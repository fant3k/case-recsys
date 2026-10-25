import unittest
import numpy as np
from recsys.data import Sequences
from recsys.reranker import FEATURES, global_item_stats, user_candidates


def make(times, items, ratios):
    return Sequences([{'uid': 1, 'timestamp': times, 'item_id': items, 'played_ratio_pct': ratios},
                      {'uid': 2, 'timestamp': [1, 2], 'item_id': [40, 50], 'played_ratio_pct': [90, 90]}],
                     [10, 20, 30, 40, 50, 60], 100, 4)


class RerankerChecks(unittest.TestCase):
    def setUp(self):
        self.times = [1, 2, 2, 5, 8, 9]
        self.items = [10, 20, 10, 30, 10, 999]
        self.ratios = [100, 40, 80, 60, 20, 90]
        self.logp = np.log(np.array([.05, .1, .15, .2, .3, .2]))

    def features(self, data, cutoff):
        user = data.users[0]
        end = int(np.searchsorted(user['times'], cutoff))
        stats = global_item_stats(data, cutoff, 6)
        return user_candidates(user, end, self.logp, cutoff, stats, np.linspace(0, 1, 6), top_gpt=2, top_full=2, context=3)

    def test_appending_future_events_does_not_change_features(self):
        base = self.features(make(self.times, self.items, self.ratios), 8)
        later = self.features(make(self.times + [12, 15], self.items + [60, 20], self.ratios + [90, 90]), 8)
        np.testing.assert_array_equal(base[0], later[0])
        np.testing.assert_array_equal(base[1], later[1])

    def test_counts_recency_and_user_item_ratio(self):
        cand, rows = self.features(make(self.times, self.items, self.ratios), 8)
        col = {f: j for j, f in enumerate(FEATURES)}
        # history before 8: 10,20,10,30 ; recent context of 3 events: 20,10,30 ; GPT top2: items 50,40
        self.assertEqual(sorted(cand.tolist()), [0, 1, 2, 3, 4])
        r = {int(c): rows[j] for j, c in enumerate(cand)}
        self.assertEqual(r[0][col['count_recent128']], 1)
        self.assertAlmostEqual(r[0][col['log_count_full']], np.log1p(2), places=5)
        self.assertAlmostEqual(r[0][col['log_seconds_since_last']], np.log1p(8 - 2), places=5)
        self.assertAlmostEqual(r[0][col['user_item_mean_ratio']], .9, places=5)
        self.assertEqual(r[0][col['user_item_positive_count']], 2)
        self.assertEqual(r[2][col['recent_position_from_end']], 0)
        self.assertEqual(r[4][col['never_seen']], 1)
        self.assertTrue(np.isnan(r[4][col['user_item_mean_ratio']]))

    def test_global_stats_use_only_past(self):
        data = make(self.times, self.items, self.ratios)
        count, count7, _ = global_item_stats(data, 3, 6)
        self.assertEqual(count[:5].tolist(), [2, 1, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
