$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskName = 'Debug-FullSFT-Reg-20261022-01'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'This run task already exists; do not start duplicate jobs.'
}
if (Test-Path 'E:\Projects\Debug\artifacts\fullsft-reg-20261022-01') {
    throw 'Run output already exists; preserving it.'
}
$action = New-ScheduledTaskAction -Execute 'E:\Projects\Debug\.venv\Scripts\python.exe' -Argument '-u -m recsys.full_sft --out artifacts/fullsft-reg-20261022-01 --seconds-per-arm 600 --eval-every-seconds 150 --arms full:0.1:0.01,full:0.3:0.01,full:0.3:0.1,full:0.5:0.1' -WorkingDirectory 'E:\Projects\Debug'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 70) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
