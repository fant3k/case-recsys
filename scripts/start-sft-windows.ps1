$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskName = 'Debug-SFT-20261021-01'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'This run task already exists; do not start duplicate jobs.'
}
if (Test-Path 'E:\Projects\Debug\artifacts\sft-20261021-01') {
    throw 'Run output already exists; preserving it.'
}
$action = New-ScheduledTaskAction -Execute 'E:\Projects\Debug\.venv\Scripts\python.exe' -Argument '-u -m recsys.train_sft --out artifacts/sft-20261021-01 --steps 8000 --batch-size 128 --eval-every 1000 --max-seconds 1200 --patience 3 --min-steps 4000' -WorkingDirectory 'E:\Projects\Debug'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 25) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State | ConvertTo-Json -Compress
