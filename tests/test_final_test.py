import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from recsys import final_test
from recsys.final_test import FINAL_MARKER, claim_final, decisions, interval, paired, rerank_logp
from recsys.reward_v2 import pair_accuracy, per_user_pair_accuracy


class FinalTestChecks(unittest.TestCase):
    def test_interval_is_bonferroni_975_and_degenerate_for_constant(self):
        r = interval(np.full(50, 0.2))
        self.assertEqual(r["level"], 0.975)
        self.assertAlmostEqual(r["mean"], 0.2)
        self.assertAlmostEqual(r["interval"][0], 0.2)
        self.assertAlmostEqual(r["interval"][1], 0.2)
        wide = interval(np.random.default_rng(0).normal(size=400))
        narrow = interval(np.random.default_rng(0).normal(size=400), level=0.95)
        self.assertLess(wide["interval"][0], narrow["interval"][0])
        self.assertGreater(wide["interval"][1], narrow["interval"][1])

    def test_paired_margin_and_user_check(self):
        a = [{"uid": 1, "ndcg": 0.5}, {"uid": 2, "ndcg": 0.2}]
        b = [{"uid": 1, "ndcg": 0.4}, {"uid": 2, "ndcg": 0.2}]
        self.assertAlmostEqual(paired(a, b)["mean"], 0.05)
        self.assertAlmostEqual(paired(a, b, margin=0.05)["mean"], 0.35 - 0.95 * 0.3)
        self.assertEqual(paired(b, b)["interval"], [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired(a, [{"uid": 2, "ndcg": 0.1}, {"uid": 1, "ndcg": 0.1}])

    def test_decision_rules(self):
        pos, zero, neg = ({"interval": [v, v + 1]} for v in (0.01, 0.0, -0.01))
        d = decisions(pos, pos, zero)
        self.assertTrue(d["H1_rm_rerank_top50_ndcg_gain_confirmed"])
        self.assertTrue(d["H2_rl_seconds_confirmed"])  # margin bound may equal 0
        self.assertFalse(decisions(zero, pos, pos)["H1_rm_rerank_top50_ndcg_gain_confirmed"])  # strict > 0
        self.assertFalse(decisions(pos, zero, pos)["H2_rl_seconds_confirmed"])
        self.assertFalse(decisions(pos, pos, neg)["H2_rl_seconds_confirmed"])

    def test_rerank_tilts_only_top_k_and_keeps_tail_ratios(self):
        torch.manual_seed(0)
        ref = F.log_softmax(torch.randn(3, 20), -1)
        idx = ref.topk(5, -1).indices
        r = torch.rand(3, 5)
        out = rerank_logp(ref, idx, r, beta=0.03)
        self.assertTrue(torch.allclose(out.exp().sum(-1), torch.ones(3)))
        tail = torch.ones_like(ref, dtype=torch.bool).scatter_(1, idx, False)
        ratio = (out - ref)[tail].reshape(3, -1)
        self.assertTrue(torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5))
        shift = (out - ref).gather(1, idx) - r / 0.03  # log pi_A - log pi_ref = r/beta + const on top-K
        self.assertTrue(torch.allclose(shift, shift[:, :1].expand_as(shift), atol=1e-4))

    def test_final_window_needs_explicit_flag(self):
        with mock.patch.object(sys, "argv", ["final_test", "--window", "final"]):
            with self.assertRaises(SystemExit) as raised:
                final_test.main()
        self.assertIn("--open-final-test", str(raised.exception))

    def test_final_out_override_refused_before_any_claim(self):
        argv = ["final_test", "--window", "final", "--open-final-test", "--out", "artifacts/elsewhere"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(final_test, "claim_final") as claim:
            with self.assertRaises(SystemExit):
                final_test.main()
        claim.assert_not_called()

    def test_missing_input_fails_before_the_one_time_claim(self):
        argv = ["final_test", "--window", "final", "--open-final-test"]
        with tempfile.TemporaryDirectory() as root, mock.patch.object(final_test, "ROOT", Path(root)), \
                mock.patch.object(sys, "argv", argv), mock.patch.object(final_test, "claim_final") as claim:
            with self.assertRaises(FileNotFoundError):
                final_test.main()
        claim.assert_not_called()

    def test_final_claim_is_once_only_and_independent_of_out(self):
        with tempfile.TemporaryDirectory() as root:
            claim_final(root, "artifacts/a")
            self.assertTrue((Path(root) / FINAL_MARKER).exists())
            with self.assertRaises(SystemExit):
                claim_final(root, "artifacts/b")

    def test_pair_accuracy_keeps_exact_50_point_float32_pairs(self):
        labels = np.array([0.7, 0.2, 0.69, 0.2], dtype=np.float32)
        uids = np.array([1, 1, 2, 2])
        r = pair_accuracy(np.array([0.9, 0.1, 0.9, 0.1]), labels, uids)
        self.assertEqual(r["pairs"], 1)  # user 1 at exactly 50 points kept, user 2 at 49 points not
        self.assertEqual(r["users_with_pairs"], 1)
        self.assertEqual(per_user_pair_accuracy(np.array([0.9, 0.1, 0.9, 0.1]), labels, uids), {1: 1.0})


if __name__ == "__main__":
    unittest.main()
