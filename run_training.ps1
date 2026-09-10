# Overnight training run. From the project dir:  powershell -ExecutionPolicy Bypass -File run_training.ps1
# Resumable: if it dies, run the same command again — it picks up from checkpoints/last.pt.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\train_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")

$resume = if (Test-Path "checkpoints\last.pt") { "auto" } else { "" }
Write-Host "logging to $log  (resume=$resume)"

python -u train.py --data augmented_resized_V2 --epochs 20 --batch-size 16 --workers 8 --resume "$resume" 2>&1 | Tee-Object $log

if ($LASTEXITCODE -eq 0) {
    python -u evaluate.py --data augmented_resized_V2 --split test --ckpt checkpoints\best.pt 2>&1 | Tee-Object -Append $log
}
Write-Host "done — see $log, checkpoints\best.pt, results\confusion_matrix_test.png"
