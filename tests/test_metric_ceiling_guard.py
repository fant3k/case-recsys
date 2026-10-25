import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recsys import metric_ceiling


class MetricCeilingGuard(unittest.TestCase):
    def test_refuses_to_read_windows_that_are_not_opened_yet(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(metric_ceiling, "ROOT", Path(root)), \
                mock.patch.object(sys, "argv", ["metric_ceiling"]):
            (Path(root) / "research/results/runs").mkdir(parents=True)
            with self.assertRaises(SystemExit) as stop:
                metric_ceiling.main()
        self.assertIn("opened only after", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
