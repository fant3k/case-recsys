"""Command-line and packaging contracts that a clean checkout relies on."""
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def run_with_argv(function, *argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["prog", *argv]), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            function()
        except SystemExit as stop:
            return stop.code, stdout.getvalue() + stderr.getvalue()
    return None, stdout.getvalue() + stderr.getvalue()


class SourceManifest(unittest.TestCase):
    def test_computed_locally_when_transfer_manifest_is_missing(self):
        from recsys import pilot
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "recsys").mkdir()
            (root / "recsys" / "a.py").write_text("x = 1\n")
            (root / "research" / "protocol").mkdir(parents=True)
            (root / "research" / "protocol" / "p.json").write_text("{}")
            (root / "recsys" / "__pycache__").mkdir()
            (root / "recsys" / "__pycache__" / "a.pyc").write_bytes(b"junk")
            with mock.patch.object(pilot, "ROOT", root):
                manifest = pilot.source_manifest()
        self.assertEqual(sorted(manifest), ["recsys/a.py", "research/protocol/p.json"])
        self.assertEqual(len(manifest["recsys/a.py"]), 64)

    def test_transfer_manifest_wins_when_present(self):
        from recsys import pilot
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "transfer-manifest.json").write_text(json.dumps({"recsys/a.py": "abc"}))
            with mock.patch.object(pilot, "ROOT", root):
                self.assertEqual(pilot.source_manifest(), {"recsys/a.py": "abc"})


class HelpIsSafe(unittest.TestCase):
    def test_positional_output_clis_do_not_create_a_help_folder(self):
        for name in ("multistep", "pairacc_recheck", "review_checks"):
            module = importlib.import_module(f"recsys.{name}")
            with self.subTest(module=name):
                code, text = run_with_argv(module.main, "--help")
                self.assertEqual(code, 0)
                self.assertIn("usage", text)
                self.assertFalse((ROOT / "--help").exists())
                self.assertFalse((ROOT / "artifacts" / "--help").exists())

    def test_option_free_clis_show_help_without_running(self):
        for name in ("metric_ceiling", "a_precision_check", "actr", "confirmation", "rl_comparison"):
            module = importlib.import_module(f"recsys.{name}")
            with self.subTest(module=name):
                code, text = run_with_argv(module.main, "--help")
                self.assertEqual(code, 0)
                self.assertIn("usage", text)

    def test_exploratory_help_does_not_raise_on_percent_sign(self):
        from recsys import exploratory
        code, text = run_with_argv(exploratory.main, "--help")
        self.assertEqual(code, 0)
        self.assertIn(">50%", text)


class ExploratoryDispatch(unittest.TestCase):
    def test_e2_options_reach_the_e2_parser(self):
        from recsys import exploratory
        with mock.patch.object(exploratory, "main_e2", return_value="e2 ran") as e2, \
                mock.patch.object(sys, "argv", ["prog", "--experiment", "e2", "--beta", "0.2", "--horizon", "3"]):
            result = exploratory.main()
        self.assertEqual(result, "e2 ran")
        e2.assert_called_once()


if __name__ == "__main__":
    unittest.main()
