# Training run: full fine-tune with discriminative LR, then test-set eval.
#   fresh:   powershell -ExecutionPolicy Bypass -File run_training.ps1
#   resume:  powershell -ExecutionPolicy Bypass -File run_training.ps1 -Resume
# NOTE: NordVPN Threat Protection must be OFF or the dataloader crawls (~1 file/s).
param([switch]$Resume)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\finetune_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")

if (-not $Resume -and (Test-Path "checkpoints_ft")) {
    $bak = "checkpoints_ft.bak_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    Rename-Item "checkpoints_ft" $bak
    Write-Host "moved stale checkpoints_ft -> $bak (fresh run)"
}
$resumeArg = if ($Resume) { "auto" } else { "" }
Write-Host "logging to $log  (resume=$resumeArg)"

python -u train.py --data augmented_resized_V2 --epochs 15 --batch-size 16 --workers 8 `
    --no-freeze --lr 3e-4 --backbone-lr 3e-5 --out checkpoints_ft --resume "$resumeArg" 2>&1 | Tee-Object $log

if ($LASTEXITCODE -eq 0) {
    python -u evaluate.py --data augmented_resized_V2 --split test --ckpt checkpoints_ft\best.pt 2>&1 | Tee-Object -Append $log
}
Write-Host "done — see $log, checkpoints_ft\best.pt, results\confusion_matrix_test.png"
