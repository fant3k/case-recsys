$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\review-fixes-20261025.log'
$steps = @(
    @{ name = 'e1_sft_control'; args = @('-u', '-m', 'recsys.exploratory', '--experiment', 'e1', '--arms', 'sft_next,sft_positive', '--out', 'artifacts/exploratory-e1-sft-20261025-01') },
    @{ name = 'pairacc_sessions'; args = @('-u', '-m', 'recsys.pairacc_recheck', 'artifacts/pairacc-recheck-20261025-01') },
    @{ name = 'multistep_gap'; args = @('-u', '-m', 'recsys.multistep', 'artifacts/multistep-20261025-01') },
    @{ name = 'review_checks_gap'; args = @('-u', '-m', 'recsys.review_checks', 'artifacts/review-checks-20261025-01') },
    @{ name = 'e2_fixed'; args = @('-u', '-m', 'recsys.exploratory', '--experiment', 'e2', '--out', 'artifacts/exploratory-e2-20261025-01') }
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
