# Training run: full fine-tune with discriminative LR, then test-set eval.
#   fresh:   powershell -ExecutionPolicy Bypass -File run_training.ps1
#   resume:  powershell -ExecutionPolicy Bypass -File run_training.ps1 -Resume
# NOTE: NordVPN Threat Protection must be OFF or the dataloader crawls (~1 file/s).
param([switch]$Resume)
# Stop only for our own cmdlets (mkdir/rename below); native-exe stderr must NOT be fatal here -
# PS 5.1 wraps every stderr line from a native command (even a harmless HF Hub warning) into a
# terminating ErrorRecord when this is "Stop", which silently killed the whole run last time.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\finetune_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")

if (-not $Resume -and (Test-Path "checkpoints_ft")) {
    $bak = "checkpoints_ft.bak_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    Rename-Item "checkpoints_ft" $bak
    Write-Host "moved stale checkpoints_ft -> $bak (fresh run)"
}
Write-Host "logging to $log  (resume=$Resume)"

$trainArgs = @("-u", "train.py", "--data", "augmented_resized_V2", "--epochs", "15",
    "--batch-size", "16", "--workers", "6", "--no-freeze", "--lr", "3e-4",
    "--backbone-lr", "3e-5", "--out", "checkpoints_ft")
if ($Resume) { $trainArgs += @("--resume", "auto") }
# empty "" args get silently dropped when PowerShell marshals argv to a native exe, which
# broke `--resume ""` here before - never pass --resume unless there's a real value.
$ErrorActionPreference = "Continue"   # native-command stderr (e.g. HF Hub warnings) must not abort the script
& python $trainArgs 2>&1 | Tee-Object $log
$trainExit = $LASTEXITCODE

if ($trainExit -eq 0) {
    python -u evaluate.py --data augmented_resized_V2 --split test --ckpt checkpoints_ft\best.pt 2>&1 | Tee-Object -Append $log
} else {
    Write-Host "train.py exited with code $trainExit - skipping eval, see $log"
}
Write-Host "done - see $log, checkpoints_ft\best.pt, results\confusion_matrix_test.png"
