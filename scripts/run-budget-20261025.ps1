$ErrorActionPreference = 'Continue'
Set-Location 'E:\Projects\Debug'
$python = 'E:\Projects\Debug\.venv\Scripts\python.exe'
$log = 'E:\Projects\Debug\artifacts\budget-20261025.log'
$wait = 'E:\Projects\Debug\artifacts\context-20261025.log'
# docs/BUDGET-PLAN.md: starts after the context runs have released the GPU.
while (-not ((Test-Path $wait) -and (Select-String -Path $wait -Pattern '^=== (all done|FAILED)' -Quiet))) { Start-Sleep -Seconds 20 }
"=== budget start $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
& $python -u -m recsys.rl_distill --out artifacts/budget-20261025-01 --arms reverse:pool --steps 8000 --lr 1.5e-4 --eval-every 1000 *>> $log
$code = $LASTEXITCODE
"=== budget exit $code $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
if ($code -ne 0) { "=== FAILED budget $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8; exit 1 }
"=== all done $(Get-Date)" | Out-File -FilePath $log -Append -Encoding utf8
