$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\joint-20261025.log'
$wait = 'E:\Projects\Debug\artifacts\rl-quantile-20261025.log'
# docs/JOINT-PLAN.md. The GPU is shared with earlier queued runs: start after the quantile RL run has finished.
while (-not ((Test-Path $wait) -and (Select-String -Path $wait -Pattern '^=== rl_quantile exit' -Quiet))) { Start-Sleep -Seconds 20 }
$failed = @()
foreach ($k in 0, 1, 2) {
    "=== joint seed offset $k start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.exploratory --experiment e1 --arms all,joint --seed-offset $k --out "artifacts/joint-seed$k-20261025-01" *>> $log
    $code = $LASTEXITCODE
    "=== joint seed offset $k exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { $failed += $k }
}
if ($failed.Count -gt 0) {
    "=== FAILED seed offsets: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
