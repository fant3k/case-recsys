"""Download the released model weights and verify them against research/weights-manifest.json.

Files are placed at the paths the code expects (artifacts/...), so `python -m recsys.final_test --window selection`
and the other evaluation steps can run without retraining. Existing files with the right SHA256 are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/weights-manifest.json"
BASE = "https://github.com/fant3k/case-recsys/releases/download/weights-v1/"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="asset names to fetch (default: all)")
    args = parser.parse_args()
    files = json.loads(MANIFEST.read_text())["files"]
    for name, entry in files.items():
        if args.only and name not in args.only:
            continue
        target = ROOT / entry["destination"]
        if target.exists() and sha256(target) == entry["sha256"]:
            print("kept", target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(BASE + name, part)
        verify(part, entry["sha256"])
        part.replace(target)
        print("fetched", target)


if __name__ == "__main__":
    main()
