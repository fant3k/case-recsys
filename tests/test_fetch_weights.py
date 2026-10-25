import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("fetch_weights_under_test", ROOT / "scripts" / "fetch_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FetchWeights(unittest.TestCase):
    def test_manifest_is_well_formed_and_destinations_stay_inside_artifacts(self):
        files = json.loads((ROOT / "research/weights-manifest.json").read_text())["files"]
        self.assertGreaterEqual(len(files), 10)
        for name, entry in files.items():
            self.assertEqual(len(entry["sha256"]), 64, name)
            self.assertTrue(entry["destination"].startswith("artifacts/"), name)
            self.assertNotIn("..", entry["destination"], name)

    def test_verify_accepts_matching_and_refuses_changed_files(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "w.bin"
            path.write_bytes(b"weights")
            module.verify(path, module.sha256(path))
            path.write_bytes(b"changed")
            with self.assertRaises(SystemExit):
                module.verify(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
