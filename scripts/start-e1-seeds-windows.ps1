$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# E1 head with seed offsets 1-4 on the selection window (descriptive; docs/EXPLORATORY-PLAN.md, 25.10 addendum).
$taskName = 'Debug-E1Seeds-20261025-01'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'This run task already exists; do not start duplicate jobs.'
}
if (Test-Path 'E:\Projects\Debug\artifacts\exploratory-e1-seed1-20261025-01') {
    throw 'Run output already exists; preserving it.'
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File E:\Projects\Debug\scripts\run-e1-seeds.ps1' -WorkingDirectory 'E:\Projects\Debug'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
