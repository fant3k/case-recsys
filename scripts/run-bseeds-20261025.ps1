$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\bseeds-20261025.log'
$wait = 'E:\Projects\Debug\artifacts\joint-20261025.log'
# docs/BSEEDS-PLAN.md: candidate B (RL on capped seconds, beta 0.3) retrained with two more seeds; selection only.
# The GPU is shared with earlier queued runs: start after the joint-training runs have finished.
while (-not ((Test-Path $wait) -and (Select-String -Path $wait -Pattern '^=== (all done|FAILED)' -Quiet))) { Start-Sleep -Seconds 20 }
$failed = @()
foreach ($k in 1, 2) {
    "=== B seed offset $k start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.rl_v2 --out "artifacts/rl-v2-bseed$k-20261025-01" --rewards seconds --betas 0.3 --offline-arms= --seed-offset $k *>> $log
    $code = $LASTEXITCODE
    "=== B seed offset $k exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { $failed += $k }
}
"=== replay start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
& $python -u -m recsys.replay_outcome --out artifacts/replay-bseeds-selection-20261025-01 `
    --extra-policy B_seed1=artifacts/rl-v2-bseed1-20261025-01/rl_seconds_beta0.3.pt `
    --extra-policy B_seed2=artifacts/rl-v2-bseed2-20261025-01/rl_seconds_beta0.3.pt *>> $log
$code = $LASTEXITCODE
"=== replay exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
if ($code -ne 0) { $failed += 'replay' }
if ($failed.Count -gt 0) {
    "=== FAILED steps: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
