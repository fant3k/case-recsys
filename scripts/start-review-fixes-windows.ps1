$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Checks requested by the Gemini red-team review (descriptive, selection window).
$taskName = 'Debug-ReviewFixes-20261025-01'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'This run task already exists; do not start duplicate jobs.'
}
if (Test-Path 'E:\Projects\Debug\artifacts\exploratory-e1-sft-20261025-01') {
    throw 'Run output already exists; preserving it.'
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File E:\Projects\Debug\scripts\run-review-fixes-20261025.ps1' -WorkingDirectory 'E:\Projects\Debug'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
