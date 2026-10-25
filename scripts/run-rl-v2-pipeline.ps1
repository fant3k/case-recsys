$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\rl-v2-pipeline-20261023-01.log'
$status = 'E:\Projects\Debug\artifacts\rl-v2-pipeline-20261023-01.json'

# Order: main RL result first, then the two new reward models and their RL runs.
$steps = @(
    @{ name = 'tests'; args = @('-m', 'unittest', 'discover', '-s', 'tests') },
    @{ name = 'rl_completion'; args = @('-u', '-m', 'recsys.rl_v2', '--out', 'artifacts/rl-v2-20261023-01') },
    @{ name = 'rm_continuation'; args = @('-u', '-m', 'recsys.reward_v2', '--out', 'artifacts/reward-v2-continuation-20261023-01',
                                          '--steps', '4000', '--ensemble', '3', '--target', 'continuation') },
    @{ name = 'rl_continuation'; args = @('-u', '-m', 'recsys.rl_v2', '--out', 'artifacts/rl-v2-continuation-20261023-01',
                                          '--rm-dir', 'artifacts/reward-v2-continuation-20261023-01',
                                          '--rewards', 'completion,completion_lcb', '--offline-arms=', '--tag', 'continuation/') },
    @{ name = 'rm_quantile'; args = @('-u', '-m', 'recsys.reward_v2', '--out', 'artifacts/reward-v2-quantile-20261023-01',
                                      '--steps', '4000', '--ensemble', '3', '--target', 'quantile') },
    @{ name = 'rl_quantile'; args = @('-u', '-m', 'recsys.rl_v2', '--out', 'artifacts/rl-v2-quantile-20261023-01',
                                      '--rm-dir', 'artifacts/reward-v2-quantile-20261023-01',
                                      '--rewards', 'completion,completion_lcb',
                                      '--offline-arms', 'exp:quantile:0.1,exp:quantile:0.3,exp:quantile:1.0', '--tag', 'quantile/') }
)

$results = @()
foreach ($step in $steps) {
    $started = Get-Date
    "=== $($step.name) start $started" | Out-File -FilePath $log -Append -Encoding utf8
    & $python @($step.args) *>> $log
    $code = $LASTEXITCODE
    $results += @{ name = $step.name; exit_code = $code; started = $started.ToString('s'); finished = (Get-Date).ToString('s') }
    @{ status = 'running'; steps = $results } | ConvertTo-Json -Depth 4 | Out-File -FilePath $status -Encoding utf8
    if ($step.name -eq 'tests' -and $code -ne 0) {
        @{ status = 'stopped_tests_failed'; steps = $results } | ConvertTo-Json -Depth 4 | Out-File -FilePath $status -Encoding utf8
        exit 1
    }
}
$failedSteps = @($results | Where-Object { $_.exit_code -ne 0 })
if ($failedSteps.Count -gt 0) {
    @{ status = 'failed_steps'; failed = @($failedSteps | ForEach-Object { $_.name }); steps = $results } | ConvertTo-Json -Depth 4 | Out-File -FilePath $status -Encoding utf8
    exit 1
}
@{ status = 'completed'; steps = $results } | ConvertTo-Json -Depth 4 | Out-File -FilePath $status -Encoding utf8
