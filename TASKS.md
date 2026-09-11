# DR Grading — Task List (top-down)

**Goal:** a reproducible EfficientNet-B3 DR grader (grades 0–4) with QWK ≥ 0.80 on a
held-out split, Grad-CAM explainability, single-image `predict.py`, and a README that
frames the work in regulatory-validation terms.

Status key: `[x]` done · `[~]` in progress · `[ ]` todo · `[?]` decision needed

---

## 0. Environment & infra

- [x] Repo scaffold, `requirements.txt`, `.gitignore`
- [x] GPU confirmed (RTX 5060, 8 GB, CUDA 12.8, torch 2.11)
- [x] Diagnose dataloader stall → Mark-of-the-Web + Defender scanning every JPEG (~1.7 s/file)
- [x] Windows Defender folder exclusion added — verified: cold reads 1700 ms → ~30 ms/file.
      MotW `Zone.Identifier` tags still on the files but Defender no longer scans them; the
      bulk `Unblock-File` pass was abandoned as unnecessary.
- [x] **Root cause of dataloader stall: NordVPN Threat Protection (`mshield` filter driver)
      scanned every content read by `python.exe` at ~1 file/s, no parallelism.** Not Defender
      (its path exclusion was active but irrelevant). MotW-tag strip did NOT help. Turning the
      NordVPN Threat Protection service off = full fix: untouched cold reads 1/s → 2,665/s.
      MotW tags were stripped from all 143k files anyway (harmless); `mshield` ADS left as-is.
      NOTE: re-enable NordVPN Threat Protection after training; it must be off for any run.
- [x] Calibration: ~234 img/s (batch 16, 8 workers, AMP), ~7.5 min/epoch train. GPU-bound at 93%.
- [x] Baseline full run done (frozen backbone, 20 epochs, ~3 h): best val QWK 0.701 (epoch 9),
      **test QWK 0.702**, acc 0.820, referable-DR sens 0.68 / spec 0.95.
      `checkpoints/best.pt`, `logs/train_20260910_072836.log`, `results/confusion_matrix_test.png`.
- [x] `requirements.lock.txt` written (`pip freeze`).
- [x] git repo (project-local `.git`, own remote) + GitHub `LSaiko/Reti-dia-grad` (private).
      NOTE: stray `.git` at `C:\Users\Admin\` is unrelated — recommend `rm -rf ~/.git`.
- [x] `--no-freeze` at single lr 3e-4 (batch 8): killed at epoch 6. Also plateaued QWK ~0.70
      (0.701/0.696/0.699/0.684/0.693) — 3e-4 too hot for pretrained backbone.
- [x] train.py: added `--backbone-lr` (discriminative LR, 2 param groups). committed 6b26079.
- [x] README Limitations section (delegated to sub-agent, e583494).
- [~] **Discriminative-LR rerun — running** (`logs/finetune_20260911_001703.log`).
      run_training.ps1 had two PS 5.1 bugs found+fixed while launching: an em-dash broke the
      unBOM'd file's parse (non-ASCII in .ps1), and `$ErrorActionPreference="Stop"` turned a
      harmless HF-Hub stderr warning into a script-killing error. Both fixed (d1ac056, 0689fd0).
      ```
      rm -rf checkpoints_ft
      python train.py --data augmented_resized_V2 --epochs 15 --batch-size 16 --workers 8 \
        --no-freeze --lr 3e-4 --backbone-lr 3e-5 --out checkpoints_ft
      ```
      then: `python evaluate.py --split test --ckpt checkpoints_ft/best.pt`
      ~2.5 h. NordVPN Threat Protection must be OFF for the run.

## 1. Data pipeline

- [x] `AlbFolder` (ImageFolder + Albumentations), `make_loaders`
- [x] Albumentations pipeline: Resize, CLAHE, RandomRotate90, HFlip, CoarseDropout, Normalize
- [x] Class-weight computation (`sklearn` balanced)
- [x] **Split-leakage check.** `augmented_resized_V2` has ~8% leakage (286 source images with
      augmented variants in both train & val, 281 train & test). `dr_unified_v2` is clean.
      → chose `augmented_resized_V2` (fits in 16 GB RAM cache; dr_unified full-res does not)
      and de-leak in code: `make_loaders(deleak=True)` drops val files whose `base_id` is in
      train, and restricts val to un-augmented `-600` originals.
- [x] Real counts recorded: train 115,241 (0:55k 1:18k 2:24k 3:8k 4:9k) ·
      val 8,730 de-leaked originals (6715/399/1333/91/192) · class weights [.42,1.25,.95,2.90,2.43]
- [x] Dataset is NOT APTOS-3662 — it's merged APTOS + EyePACS (~90k source images). README must be fixed.
- [?] Circular-crop / retina-border removal preprocessing — images vary a lot in framing.
      Decide add (helps) vs skip (simpler). Default: skip for v1.
- [?] Move CLAHE offline (precompute) if it shows up as a CPU bottleneck (calibration will tell)

## 2. Model

- [x] `timm` EfficientNet-B3, 5-class softmax
- [x] Layer freezing (last 2 MBConv stages + head), `--no-freeze` escape hatch
- [x] Grad-CAM target layer helper (`conv_head`)
- [ ] Confirm 300×300 input + ImageNet normalization is what the pretrained weights expect
- [?] Ordinal head (cumulative-logits / CORAL) — README says softmax is enough for v1.
      Revisit only if QWK stalls below target. Keep as a documented alternative.

## 3. Training

- [x] Loop: class-weighted CrossEntropyLoss, AdamW on trainable params
- [x] Mixed precision (AMP), `--no-amp` escape hatch
- [x] Throughput logging (img/s per 50 batches + per-epoch)
- [x] Best-checkpoint save keyed on val QWK (weights + class names + img_size)
- [x] Cosine LR schedule (`CosineAnnealingLR`, T_max = epochs)
- [x] Seed everything (random, numpy, torch, cuda); cudnn.benchmark on
- [x] `last.pt` every epoch (model+opt+sched+scaler+epoch) + `--resume auto` for interrupted runs
- [x] `run_training.ps1` — one-shot train → test-eval, logs to `logs/`
- [~] Full training run (20 epochs) — RUNNING (pid 1633), epoch 1 = calibration
- [~] Early stopping / patience — skipped; cosine + fixed 20 epochs + resumable is enough for v1

## 4. Evaluation & metrics

- [x] Per-epoch QWK (quadratic-weighted Cohen's kappa — the APTOS metric)
- [x] Per-epoch confusion matrix + per-class precision/recall/F1 (printed)
- [x] `evaluate.py`: load a checkpoint, run test/val, print QWK + CM + per-class report
- [x] Save confusion matrix PNG (`results/confusion_matrix_<split>.png`)
- [x] Referable-DR (grade ≥ 2) sensitivity/specificity in `evaluate.py`
- [ ] Run `evaluate.py --split test` after training, paste numbers into README
- [?] Temperature scaling for confidence calibration (README mentions it) — small
      post-hoc step on the val split; add if `predict.py` confidences look over-confident

## 5. Explainability (Grad-CAM)

- [x] `GradCAM` class (hooks on last conv, GAP gradients, ReLU, upsample, normalize)
- [x] `overlay_gradcam(image_path, model, gradcam, class_idx)` → jet-blended overlay
- [x] `batch_overlay(folder, ...)` → writes `results/`
- [x] Self-check (`test_gradcam.py`) on the CAM math
- [ ] Run `batch_overlay` on a sample of real fundus images once a checkpoint exists
- [ ] Hand-review ~20 overlays: is the model attending to lesions (hemorrhages,
      microaneurysms, exudates, neovascularization) vs. artifacts (border, flare, JPEG blocks)?
- [ ] Put 3–4 representative overlays (one per non-zero grade) in the README

## 6. Inference deliverable

- [x] `predict.py`: single image → grade + confidence + per-class probs + heatmap PNG
- [ ] End-to-end test `predict.py` against `best.pt` on a held-out image
- [ ] Handle the no-checkpoint / bad-path cases with a clear error message
- [?] Tiny CLI batch mode (`predict.py folder/`) — only if it's actually wanted

## 7. Documentation & regulatory framing

- [x] README rewritten: real dataset (APTOS+EyePACS), de-leak + originals-only split method,
      MotW/Windows repro note, training details, results table set to "pending"
- [ ] Fill in the **real** QWK / accuracy / per-class / referable-DR numbers after training
- [ ] Add the confusion-matrix PNG and Grad-CAM example images
- [ ] Add a LIMITATIONS section: merged dataset, no external validation, no subgroup
      analysis, offline-augmentation caveat, not a cleared device

## 8. Stretch (only if time / interest)

- [ ] EyePACS (88k) as an external validation set — the real generalization test
- [ ] Ordinal-regression head A/B vs. softmax, compare QWK
- [ ] Test-time augmentation (TTA) for the final prediction
- [ ] ONNX export + a note on inference latency

---

## Suggested order

1. Finish infra (0): unblock completes → calibration → lock the plan
2. Data integrity (1.4 split-leakage check) — **do this before any full run**, it decides
   whether the numbers mean anything
3. Full training run (3) + test-set eval (4)
4. Grad-CAM review on real images (5) + inference test (6)
5. Backfill README with real numbers and images (7)
