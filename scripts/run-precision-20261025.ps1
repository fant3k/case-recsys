$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\precision-20261025.log'
# docs/PRECISION-PLAN.md: A = FP32 training and FP32 reward cache, B = FP16 with dynamic loss scaling.
$common = @('-u', '-m', 'recsys.rl_distill', '--arms', 'reverse:pool,forward:pool', '--grad-diagnostic', '4')
$steps = @(
    @{ name = 'A_fp32'; args = $common + @('--precision', 'fp32', '--cache-fp32', '--out', 'artifacts/precision-fp32-20261025-01') },
    @{ name = 'B_fp16scaled'; args = $common + @('--precision', 'fp16scaled', '--out', 'artifacts/precision-fp16scaled-20261025-01') }
)
$failed = @()
foreach ($step in $steps) {
    "=== $($step.name) start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python @($step.args) *>> $log
    $code = $LASTEXITCODE
    "=== $($step.name) exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { $failed += $step.name }
}
if ($failed.Count -gt 0) {
    "=== FAILED steps: $($failed -join ', ') $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
