$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\context-20261025.log'
# docs/CONTEXT-PLAN.md: full+pps+fb, 600 s per arm, context 256 and 512; batch 32 for 512 only if batch 64 fails (memory).
$failed = @()
foreach ($size in 256, 512) {
    "=== context $size start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -u -m recsys.full_sft --out "artifacts/context$size-20261025-01" --seconds-per-arm 600 --eval-every-seconds 150 --arms 'full+pps+fb:0.1:0.01' --context-size $size *>> $log
    $code = $LASTEXITCODE
    "=== context $size exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0 -and $size -eq 512) {
        "=== context 512 retry with batch 32 $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
        & $python -u -m recsys.full_sft --out "artifacts/context512-b32-20261025-01" --seconds-per-arm 600 --eval-every-seconds 150 --arms 'full+pps+fb:0.1:0.01' --context-size 512 --batch-size 32 *>> $log
        $code = $LASTEXITCODE
        "=== context 512 batch32 exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    }
    if ($code -ne 0) { $failed += $size }
}
if ($failed.Count -gt 0) {
    "=== FAILED contexts: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
