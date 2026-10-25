$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\joint-rows-20261025.log'
# docs/SFTKL-PLAN.md: one run per seed, fixed lambdas, hard stop.
$failed = @()
foreach ($k in 0, 1, 2) {
    "=== joint-rows seed offset $k start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.exploratory --experiment e1 --arms 'all,joint,sft_kl:0.1,sft_kl:1,sft_kl:10' --seed-offset $k --out "artifacts/joint-rows-seed$k-20261025-01" *>> $log
    $code = $LASTEXITCODE
    "=== joint-rows seed offset $k exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { $failed += $k }
}
if ($failed.Count -gt 0) {
    "=== FAILED seed offsets: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
