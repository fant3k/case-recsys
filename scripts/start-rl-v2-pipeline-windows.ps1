$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskName = 'Debug-RLv2-Pipeline-20261023-01'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'This run task already exists; do not start duplicate jobs.'
}
foreach ($path in 'artifacts\rl-v2-20261023-01', 'artifacts\reward-v2-continuation-20261023-01', 'artifacts\rl-v2-continuation-20261023-01',
                  'artifacts\reward-v2-quantile-20261023-01', 'artifacts\rl-v2-quantile-20261023-01', 'artifacts\rl-v2-pipeline-20261023-01.json') {
    if (Test-Path (Join-Path 'E:\Projects\Debug' $path)) { throw "Run output already exists; preserving it: $path" }
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File E:\Projects\Debug\scripts\run-rl-v2-pipeline.ps1' -WorkingDirectory 'E:\Projects\Debug'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 5) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
