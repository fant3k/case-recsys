import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

import sys
from unittest import mock

from recsys import new_cohort
from recsys.new_cohort import MARKER, MARKER_DAY2, check_mode, claim, cohort_sequences, decisions, decisions_day2


class NewCohortChecks(unittest.TestCase):
    def test_only_two_modes_and_explicit_flag(self):
        check_mode("original", "selection", False, None)
        check_mode("new", "final", True, None)
        for args in [("original", "final", False, None), ("new", "selection", True, None),
                     ("new", "final", False, None), ("new", "final", True, "artifacts/x")]:
            with self.assertRaises(SystemExit):
                check_mode(*args)

    def test_second_day_modes_and_separate_marker(self):
        check_mode("original", "confirmation", False, "artifacts/x")
        check_mode("new", "confirmation", True, None)
        for args in [("new", "confirmation", False, None), ("new", "confirmation", True, "artifacts/x")]:
            with self.assertRaises(SystemExit):
                check_mode(*args)
        self.assertNotEqual(MARKER, MARKER_DAY2)
        with tempfile.TemporaryDirectory() as root:
            claim(root, "a")
            claim(root, "b", MARKER_DAY2)  # the final-window claim does not block the second day
            with self.assertRaises(SystemExit):
                claim(root, "c", MARKER_DAY2)

    def test_day2_decisions_strict(self):
        d = decisions_day2({"interval": [0.0005, 0.01]}, {"interval": [-0.001, 0.01]})
        self.assertTrue(d["N3a_A_beats_reference_on_new_users_other_day"])
        self.assertFalse(d["N3b_E1_head_beats_reference_on_new_users_other_day"])

    def test_claim_once(self):
        with tempfile.TemporaryDirectory() as root:
            claim(root, "a")
            self.assertTrue((Path(root) / MARKER).exists())
            with self.assertRaises(SystemExit):
                claim(root, "b")

    def test_overlap_with_original_is_refused(self):
        rows = {"uid": [1, 2], "timestamp": [[10, 20], [5, 30]], "item_id": [[7, 8], [8, 9]], "is_organic": [[0, 1], [1, 0]],
                "played_ratio_pct": [[100, 40], [60, 80]], "track_length_seconds": [[200, 180], [180, 240]]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "c.parquet"
            pl.DataFrame(rows).write_parquet(path)
            seq = cohort_sequences(path, np.array([7, 8, 9]), 25, 128, [3, 4])
            self.assertEqual(sorted(u["uid"] for u in seq.users), [1, 2])
            with self.assertRaises(ValueError):
                cohort_sequences(path, np.array([7, 8, 9]), 25, 128, [2, 3])

    def test_decisions_strict(self):
        d = decisions({"interval": [0.001, 0.01]}, {"interval": [0.0, 0.01]})
        self.assertTrue(d["N1_A_beats_reference_on_new_users"])
        self.assertFalse(d["N2_E1_head_beats_reference_on_new_users"])

    def test_missing_file_fails_before_claim(self):
        argv = ["new_cohort", "--cohort", "new", "--window", "final", "--open-new-cohort"]
        with tempfile.TemporaryDirectory() as root, mock.patch.object(new_cohort, "ROOT", Path(root)), \
                mock.patch.object(sys, "argv", argv), mock.patch.object(new_cohort, "claim") as claimed:
            with self.assertRaises(FileNotFoundError):
                new_cohort.main()
        claimed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
