"""Project-scoped SSH/SCP helper; credentials remain in the Mac SSH directory."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HOST = os.environ.get("DEBUG_WINDOWS_HOST", "user@windows-host")
SSH = [
    "-i", str(Path.home()/".ssh/debug_windows_ed25519"),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile="+str(Path.home()/".ssh/debug_windows_known_hosts"),
    "-o", "ConnectTimeout=8",
]


def powershell(script):
    script = "$ProgressPreference = 'SilentlyContinue'\n" + script
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.call(["ssh", *SSH, HOST, "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand "+encoded])


def sync():
    paths = [p for p in (ROOT/"requirements-windows.txt", ROOT/"README.md", ROOT/"CLAUDE.md") if p.exists()]
    for directory in ["recsys", "tests", "scripts", "docs", "research/protocol", "research/results"]:
        paths.extend(p for p in (ROOT/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.append(ROOT/"research/weights-manifest.json")
    paths.append(ROOT/"data/processed/yambda-1000-users-full-histories.parquet")
    cohort = ROOT/"data/processed/yambda-new-cohort-1000.parquet"  # fresh users, docs/NEW-COHORT-PLAN.md
    if cohort.exists():
        paths.append(cohort)
    paths.extend(ROOT/"research/selection"/name for name in ("select_new_cohort.py", "new-cohort-selection.json"))
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tempfile.TemporaryDirectory(prefix="debug-windows-") as folder:
        archive = Path(folder)/"project-transfer.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            for p in paths:
                z.write(p, str(p.relative_to(ROOT)))
            z.writestr("transfer-manifest.json", json.dumps(manifest, indent=2))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if powershell("$ErrorActionPreference='Stop'\nNew-Item -ItemType Directory -Force 'E:\\Projects\\Debug\\.transfer' | Out-Null"):
            raise SystemExit("Remote directory creation failed")
        subprocess.run(["scp", *SSH, str(archive), HOST+":E:/Projects/Debug/.transfer/project-transfer.zip"], check=True)
        script = f"""
$ErrorActionPreference='Stop'
$archive='E:\\Projects\\Debug\\.transfer\\project-transfer.zip'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower() -ne '{digest}') {{ throw 'Archive hash mismatch' }}
Expand-Archive -LiteralPath $archive -DestinationPath 'E:\\Projects\\Debug' -Force
$manifest=Get-Content 'E:\\Projects\\Debug\\transfer-manifest.json' -Raw | ConvertFrom-Json
foreach ($entry in $manifest.PSObject.Properties) {{
  $path=Join-Path 'E:\\Projects\\Debug' $entry.Name
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() -ne $entry.Value) {{ throw ('File hash mismatch: '+$entry.Name) }}
}}
Write-Output ('TRANSFER_HASHES_PASS: '+@($manifest.PSObject.Properties).Count)
"""
        raise SystemExit(powershell(script))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sync", "exec", "fetch"])
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    if args.mode != "sync" and not args.value:
        parser.error("exec/fetch require a command, script file or relative file path")
    if args.mode == "sync":
        sync()
    elif args.mode == "exec":
        try:
            is_file = Path(args.value).is_file()
        except OSError:
            is_file = False
        script = Path(args.value).read_text() if is_file else args.value
        raise SystemExit(powershell(script))
    else:
        relative = Path(args.value)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("Use a project-relative path")
        local = ROOT/relative
        local.parent.mkdir(parents=True, exist_ok=True)
        remote = HOST+":E:/Projects/Debug/"+relative.as_posix()
        subprocess.run(["scp", *SSH, remote, str(local)], check=True)


if __name__ == "__main__":
    main()
