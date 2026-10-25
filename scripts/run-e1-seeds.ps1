$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\e1-seeds-20261025.log'
$failed = @()
foreach ($k in 1, 2, 3, 4) {
    "=== seed offset $k start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.exploratory --experiment e1 --seed-offset $k --out "artifacts/exploratory-e1-seed$k-20261025-01" *>> $log
    $code = $LASTEXITCODE
    "=== seed offset $k exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { $failed += $k }
}
if ($failed.Count -gt 0) {
    "=== FAILED seed offsets: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
