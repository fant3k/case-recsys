$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\rl-quantile-20261025.log'
$wait = 'E:\Projects\Debug\artifacts\precision-20261025.log'
# The GPU is shared with the precision runs (docs/PRECISION-PLAN.md): start only after they have finished.
while (-not ((Test-Path $wait) -and (Select-String -Path $wait -Pattern '^=== (all done|FAILED)' -Quiet))) { Start-Sleep -Seconds 20 }
"=== rl_quantile start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
& $python -u -m recsys.rl_v2 --out artifacts/rl-v2-quantile-20261025-01 --rm-dir artifacts/reward-v2-quantile-20261023-01 `
    --rewards completion --betas 0.03,0.1,0.3,1.0 --offline-arms 'sft,exp:quantile:0.1,exp:quantile:0.3,exp:quantile:1.0' --tag quantile/ *>> $log
$code = $LASTEXITCODE
"=== rl_quantile exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
