$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'E:\Projects\Debug'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& '.\.venv\Scripts\python.exe' -m recsys.pilot --out artifacts/pilot-20261021-01 --sft-steps 80 --rm-steps 200 --rl-steps 40 --batch-size 16 --reward-users 64 --max-seconds 900
exit $LASTEXITCODE
