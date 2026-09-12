# DR Grading — Task List (top-down)

**Goal:** a reproducible EfficientNet-B3 DR grader (grades 0–4) with QWK ≥ 0.80 on a
held-out split, Grad-CAM explainability, single-image `predict.py`, and a README that
frames the work in regulatory-validation terms.

**STUDY WRAPPED UP.** The 0.80 QWK target was not reached (best: 0.723, unboosted
fine-tune); a documented trade-off exists between that and grade-1 detection instead
(0.710 QWK / 0.112 grade-1 F1, the boosted model, now the default). See README
Conclusion + Future Work for the final writeup and ranked next steps. This file is now a
project history / reference, not an active plan.

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
- [x] **Discriminative-LR run — DONE.** Took 4 launch attempts, 3 real bugs found+fixed along
      the way: an em-dash broke the unBOM'd .ps1's parse; `$ErrorActionPreference="Stop"` turned
      a harmless HF-Hub stderr warning into a script-killing error; both loaders sharing
      `--workers` gave 16 persistent worker processes (8 train + 8 val) → `MemoryError` in a
      worker mid-run, main process hung ~15.5h with zero epochs done before I caught it and
      killed it. Fixed: val workers capped to 2, non-persistent (a53e0b5). Final successful run:
      15 epochs, 542s/epoch (213 img/s), `logs/finetune_20260911_155432.log`.
      **Result: best val QWK 0.7173 (epoch 5), test QWK 0.7234, acc 0.8401, referable-DR
      sens 0.698 / spec 0.954.** Improved over baseline (0.702) but grade-1 (Mild) recall
      *dropped* 0.08→0.03 — fine-tuning didn't fix the hard class, it made it worse. Below the
      0.80 target. `checkpoints_ft/best.pt`, `results/confusion_matrix_test.png`.
- [x] **Targeted grade-1 fix — DONE.** `--boost-class 1 --boost-factor 4.0` (class weight
      0.42→4.99, highest of all 5), same discriminative-LR config, 15 epochs,
      `checkpoints_grade1fix/best.pt`, `logs/grade1fix_20260911_205609.log`.
      **Result: test QWK 0.7097, acc 0.8211, grade-1 F1 0.04→0.112 (nearly 3x the unboosted
      fine-tune, beats baseline's 0.09 too), referable-DR sens 0.674 / spec 0.960.**
      Real fix on the target metric, at a real cost: -0.013 QWK and -0.024 sensitivity vs. the
      unboosted fine-tune. Unstable early (epoch-1 val QWK cratered to 0.52, grade-1 recall
      spiked to 0.64 then settled to ~0.10-0.15 by epoch 5+) before reaching a stable trade-off.
      Live-monitored via a published Artifact dashboard (progress bar, per-epoch QWK chart vs.
      reference lines, per-class F1) updated after each epoch during the run.
      **Still below the 0.80 target — three models now exist, none reaching it: baseline
      (0.702, weak grade-1), unboosted fine-tune (0.723, worst grade-1), boosted fine-tune
      (0.710, best grade-1). No further free action — remaining options (focal loss,
      oversampling, ordinal head, TTA) all need another GPU run and don't have an obvious
      reason to beat this trade-off rather than just move it.**

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
- [x] Dataset is NOT APTOS-3662 — it's merged APTOS + EyePACS (~90k source images). README fixed.
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
- [x] Discriminative LR (`--backbone-lr`) — see section 0. Two full runs done: frozen baseline
      (QWK 0.702) and full fine-tune (QWK 0.723). Both checked in.
- [x] Early stopping / patience — skipped; cosine + fixed epoch budget + resumable was enough

## 4. Evaluation & metrics

- [x] Per-epoch QWK (quadratic-weighted Cohen's kappa — the APTOS metric)
- [x] Per-epoch confusion matrix + per-class precision/recall/F1 (printed)
- [x] `evaluate.py`: load a checkpoint, run test/val, print QWK + CM + per-class report
- [x] Save confusion matrix PNG (`results/confusion_matrix_<split>.png`)
- [x] Referable-DR (grade ≥ 2) sensitivity/specificity in `evaluate.py`
- [x] Ran `evaluate.py --split test` for both models, numbers are in README (f7fe7c3)
- [?] Temperature scaling for confidence calibration (README mentions it) — small
      post-hoc step on the val split; add if `predict.py` confidences look over-confident

## 5. Explainability (Grad-CAM)

- [x] `GradCAM` class (hooks on last conv, GAP gradients, ReLU, upsample, normalize)
- [x] `overlay_gradcam(image_path, model, gradcam, class_idx)` → jet-blended overlay
- [x] `batch_overlay(folder, ...)` → writes `results/`
- [x] Self-check (`test_gradcam.py`) on the CAM math
- [x] Ran `batch_overlay` on 10 test images (2/grade) with the fine-tuned model
- [x] Hand-reviewed the samples: **finding** — `conv_head` is 10×10 at 300px input, heatmaps
      are coarse 30×-upsampled blobs, and heat often sits near the fundus border rather than on
      vessels/lesions. Documented as an open caveat (6d547e3) — could be genuine border
      sensitivity or just the coarse resolution, not distinguished.
- [ ] **Not done: put actual overlay images in the README** (sent to user via chat, not
      committed/embedded — a private GitHub repo can embed images via `results/*.png` links,
      or upload to an `docs/img/` folder and reference by relative path)
- [ ] Follow-up on the border finding: try a shallower target layer (e.g. next-to-last block,
      higher spatial res) and re-review whether the border concentration persists

## 6. Inference deliverable

- [x] `predict.py`: single image → grade + confidence + per-class probs + heatmap PNG
- [x] End-to-end tested against `checkpoints_ft/best.pt` — correct grade, 0.972 confidence
- [x] Clean errors for missing checkpoint, missing image, and unreadable/corrupt image
      (one-line message + exit 1, no raw traceback). Tested all 3 paths + happy path (0699019).
- [?] Tiny CLI batch mode (`predict.py folder/`) — only if it's actually wanted

## 7. Documentation & regulatory framing

- [x] README rewritten: real dataset (APTOS+EyePACS), de-leak + originals-only split method,
      MotW/Windows repro note, training details, results table filled in for both models
- [x] Limitations section (7 points incl. grade-1 weakness and the Grad-CAM border finding)
- [x] Confusion matrix + one Grad-CAM overlay per grade embedded in README, committed to
      `docs/img/` (not `results/`, which stays gitignored/regenerable) (0699019)
- [ ] Add architecture/pipeline diagram (optional, portfolio polish)

## 8. Stretch (only if time / interest)

- [ ] EyePACS (88k) as an external validation set — the real generalization test
- [ ] Ordinal-regression head A/B vs. softmax, compare QWK
- [ ] Test-time augmentation (TTA) for the final prediction
- [ ] ONNX export + a note on inference latency

---

## Where things stand

Three full training runs done, all committed: frozen baseline (test QWK 0.702, grade-1 F1
0.09), full fine-tune (0.723, grade-1 F1 0.04 — capacity alone made grade-1 worse), and
fine-tune + 4x grade-1 class-weight boost (0.710, grade-1 F1 0.112 — best on the target
metric, costs 0.013 QWK vs. the unboosted fine-tune). None reaches the 0.80 target. This is
now a real trade-off between three models, not a single number to chase further without a
reason to think the next lever (focal loss, oversampling, ordinal head) would do better than
just relocate the same trade-off. Everything else in the pipeline (data, eval, Grad-CAM,
inference, docs) is functional and checked in. Live-monitored the boost run via a published
Artifact dashboard (progress bar + per-epoch chart + per-class F1, updated each epoch).
**NordVPN Threat Protection must be off before any future training run** (toggle it back on
when done — it's a real security feature, just incompatible with this dataloader).

## Next tasks

- [x] ~~Embed results in README~~ — confusion matrix + per-grade Grad-CAM gallery, `docs/img/` (0699019)
- [x] ~~`predict.py` error handling~~ — clean one-line errors, tested (0699019)
- [x] ~~Targeted grade-1 fix~~ — class-weight boost, done, see section 0 (result: 0.710 QWK / 0.112 grade-1 F1)

- [x] **Picked the default model: `checkpoints_grade1fix/best.pt`** (`DEFAULT_CKPT` in
      `predict.py`, imported into `evaluate.py` too). Chose it over the higher-QWK unboosted
      fine-tune (0.723 vs 0.710) because a screening tool that's nearly blind to Mild DR
      (grade-1 recall 0.03) is a worse default than one that trades a little QWK to actually
      see it (recall 0.111). `--ckpt checkpoints_ft/best.pt` still available for the other one.

- [x] **Grad-CAM border finding — investigated and fixed.** Re-ran the same 10 sample images
      (grade1fix checkpoint) with the target layer moved from `conv_head` (10x10) to
      `blocks[4]` (19x19, ~3.6x the cells). Consistent, visible improvement: coarse blobs that
      hugged the image edge became many small hotspots landing on the optic disc, vessel
      arcades, and scattered lesion-like points. **Changed `model.py`'s `target_layer()`
      default to `blocks[4]`**; old behavior kept as `target_layer_coarse()`. A residual
      conv-padding edge artifact remains (heat bleeding past the fundus circle into the black
      background) — a separate, lower-stakes issue, documented in README. Before/after images
      in `docs/img/`, full writeup in README Explainability + Conclusion.
- [x] **Study wrapped up.** README got a Conclusion section (what was actually learned) and a
      ranked Future Work / addendum section — top pick is ensembling the 3 existing
      checkpoints (no new training, directly targets the documented trade-off), down through
      focal loss, domain pretraining, lesion-segmentation auxiliary task, external validation,
      and ordinal regression (ranked last — the grade-1 experiments here point to a label-
      quality problem, not a loss-framing problem).

Nothing left blocking. Any further work is genuinely optional and is listed, ranked, in
README's Future Work section rather than here.
