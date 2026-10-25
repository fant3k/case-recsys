$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'E:\Projects\Debug'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

& '.\.venv\Scripts\python.exe' -m recsys.pilot --out artifacts/calibration-20261021-01 --sft-steps 4000 --rm-steps 5000 --rl-steps 200 --batch-size 16 --reward-users 64 --max-seconds 300
exit $LASTEXITCODE
