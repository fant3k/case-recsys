$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\qseeds-20261025.log'
# docs/QSEEDS-PLAN.md: RL on the quantile reward, beta 0.3, two more seeds; selection only.
$failed = 0
foreach ($k in 1, 2) {
    "=== quantile seed offset $k start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.rl_v2 --out "artifacts/rl-v2-qseed$k-20261025-01" --rm-dir artifacts/reward-v2-quantile-20261023-01 `
        --rewards completion --betas 0.3 --offline-arms= --tag quantile/ --seed-offset $k *>> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { $failed = 1 }
    "=== quantile seed offset $k exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
}
if ($failed) { "=== FAILED $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8 } else { "=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8 }
exit $failed
