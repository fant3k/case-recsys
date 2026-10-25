import unittest

from recsys.replay_outcome import liked_hits, summarize, user_outcomes


class ReplayOutcome(unittest.TestCase):
    def test_only_recommended_and_played_tracks_are_credited(self):
        items = {10: (120.0, 2, 1.5), 11: (30.0, 1, 0.25), 99: (500.0, 3, 3.0)}
        got = user_outcomes(items, [10, 11, 12])
        self.assertEqual(got, {"replay_seconds": 150.0, "replay_plays": 3, "completion_sum": 1.75})

    def test_user_without_listening_gets_zero(self):
        self.assertEqual(user_outcomes({}, [1, 2])["replay_seconds"], 0.0)

    def test_summary_is_paired_and_pools_completion(self):
        mine = [{"replay_seconds": 10.0, "replay_plays": 1, "completion_sum": 0.5}, {"replay_seconds": 0.0, "replay_plays": 0, "completion_sum": 0.0}]
        ref = [{"replay_seconds": 4.0, "replay_plays": 1, "completion_sum": 0.25}, {"replay_seconds": 2.0, "replay_plays": 1, "completion_sum": 1.0}]
        result = summarize(mine, ref)
        self.assertAlmostEqual(result["replay_seconds"]["delta_vs_reference"]["mean"], 2.0)
        self.assertAlmostEqual(result["completion_on_played_recommended_tracks"], 0.5)
        self.assertEqual(result["played_recommended_tracks"], 1)

    def test_liked_hits_counts_only_recommended_liked_tracks(self):
        self.assertEqual(liked_hits({1, 5, 9}, [1, 2, 5, 7]), 2)
        self.assertEqual(liked_hits(set(), [1, 2]), 0)

    def test_summary_includes_likes_when_present(self):
        mine = [{"replay_seconds": 0.0, "replay_plays": 0, "completion_sum": 0.0, "liked_top10": 2},
                {"replay_seconds": 0.0, "replay_plays": 0, "completion_sum": 0.0, "liked_top10": 0}]
        ref = [{"replay_seconds": 0.0, "replay_plays": 0, "completion_sum": 0.0, "liked_top10": 1}] * 2
        result = summarize(mine, ref)["liked_top10"]
        self.assertEqual(result["liked_recommended_tracks"], 2.0)
        self.assertEqual(result["users_with_a_liked_recommendation"], 1)
        self.assertAlmostEqual(result["delta_vs_reference"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
