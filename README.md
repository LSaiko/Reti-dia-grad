# Diabetic Retinopathy Grading — EfficientNet-B3

A 5-class diabetic retinopathy (DR) severity classifier for retinal fundus
photographs, with Grad-CAM explainability. Grades follow the international
clinical scale: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative.

This is a research / portfolio implementation. It is **not** a medical device
and has not been cleared or approved by any regulatory body. The regulatory
notes below describe how the technical artifacts map to design-control
expectations, not a claim of compliance.

## Results

Held-out test split (8,741 de-leaked original images), EfficientNet-B3.

| Metric | Frozen backbone (baseline) | Full fine-tune (discriminative LR) |
|---|---|---|
| Quadratic Weighted Kappa (QWK) | 0.702 | **0.723** |
| Accuracy | 0.820 | 0.840 |
| Referable-DR (grade ≥ 2) sensitivity | 0.68 | 0.698 |
| Referable-DR (grade ≥ 2) specificity | 0.95 | 0.954 |

Per-class F1 (fine-tuned): 0 No-DR 0.92 · 1 Mild 0.04 · 2 Moderate 0.62 ·
3 Severe 0.28 · 4 Proliferative 0.69. Confusion matrix:
`results/confusion_matrix_test.png`.

QWK (quadratic-weighted Cohen's kappa) is the standard DR-grading metric — it
penalizes errors by how far off the grade is. The **baseline** trains only the
last two EfficientNet stages + head; frozen ImageNet features plateau at
QWK ≈ 0.70. Unfreezing the whole backbone with a **discriminative learning
rate** (`--no-freeze --lr 3e-4 --backbone-lr 3e-5`, 15 epochs) improved QWK,
accuracy, and referable-DR sensitivity — but **made grade-1 (Mild DR) worse**,
not better: recall dropped from 0.08 to 0.03 (only ~10 of 398 mild-DR eyes
correctly identified). The extra capacity appears to have gone toward the
majority/moderate classes rather than resolving the subtle grade-0/grade-1
boundary; see Limitations.

## Dataset

A merged **APTOS 2019 + EyePACS** fundus-photo set (~90k source images graded
0–4), arranged as an `ImageFolder`:

```
<data_root>/
  train/{0,1,2,3,4}/*.jpg   # offline-augmented (multiple variants per source image)
  val/{0,1,2,3,4}/*.jpg
  test/{0,1,2,3,4}/*.jpg
```

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — 3,662 images
- [EyePACS / Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) — ~88k images

The label distribution is severely skewed: on un-augmented images **grade 0 ≈ 74%**,
grades 3–4 combined ≈ 3%. A model trained without addressing this collapses to
predicting grade 0 for everything — ~74% accuracy while being clinically useless.

**Split integrity.** The `train` set contains multiple offline-augmented copies of
each source photo. `data.py` (`make_loaders`, `make_eval_loader`) enforces two
things so metrics stay honest:
1. **de-leaking** — any val/test image whose source id also appears in `train` is
   dropped (the provided offline split shares ~8% of source images across splits);
2. **originals only** — val/test are restricted to the un-augmented image per
   source, so metrics aren't computed over correlated augmented duplicates.

### Reproducing on Windows

The dataset ships from a downloaded archive, so every file carries a
Mark-of-the-Web tag (`Zone.Identifier`). On-access malware scanners (Windows
Defender, NordVPN Threat Protection, etc.) re-scan MotW-tagged files on every
open — this throttled the dataloader to ~0.5 img/s until the tags were stripped
(`Get-ChildItem -Recurse | Unblock-File`). Strip the tags and/or add the dataset
folder to your scanner's exclusions before training.

## Approach

### Classification framing
DR grades are ordinal — grade 2 sits *between* grades 1 and 3. Two options:
5-class softmax (simple) or ordinal/cumulative-probability regression. This
implementation uses **softmax**, which is sufficient here; the QWK metric and
class weighting carry most of the ordinal signal in practice.

### Class imbalance
Class weights are computed with `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`
from the training labels and passed to `CrossEntropyLoss(weight=...)`. This
raises the loss contribution of the rare referable grades (3, 4) so the model
cannot ignore them.

### Architecture
EfficientNet-B3 via `timm`, 300×300 input, ImageNet normalization. By default
the early stages are frozen and only the last two MBConv stages plus the head
are fine-tuned (`--no-freeze` trains everything). EfficientNet gives a better
accuracy/compute trade-off than a comparable ResNet for this task.

### Augmentation
On-the-fly Albumentations pipeline (train only): `Resize`, `CLAHE` (contrast
normalization for varying fundus-camera exposure), `RandomRotate90`,
`HorizontalFlip`, `CoarseDropout` (occlusion robustness), `Normalize`. This is on
top of whatever offline augmentation the dataset already carries.

### Training
Seeded, AMP (mixed precision), AdamW, cosine LR schedule over the epoch budget.
`train.py` writes `checkpoints/last.pt` every epoch and `checkpoints/best.pt` on
val-QWK improvement; re-run with `--resume auto` to continue an interrupted run.

### Explainability — Grad-CAM
`gradcam.py` implements Grad-CAM from scratch: forward/backward hooks on the
last convolutional layer (`model.conv_head`), gradients global-average-pooled
into channel weights, weighted activation sum passed through ReLU, upsampled and
normalized to a `[0,1]` heatmap. `overlay_gradcam()` blends a jet colormap over
the original photo; `batch_overlay()` processes a folder into `results/`.

The intended use of the heatmaps is **interpretability review**: confirming that
predictions are driven by retinal features (microaneurysms, hemorrhages,
neovascularization, exudates) rather than image artifacts — lens flare, border
vignetting, JPEG blocking, or laser photocoagulation scars from prior treatment.
A model that grades correctly for the wrong reason will not generalize to a new
camera or clinic.

**Caveat on resolution.** `conv_head` outputs a 10×10 spatial map at 300×300
input, so every heatmap is a handful of coarse, blocky regions upsampled 30×
rather than lesion-tight localization. On manual review of sample overlays
across all five grades, heat is often concentrated near the fundus border
rather than clearly on vessels or lesions. That may be genuine
border/vignetting sensitivity, or may just be an artifact of using the very
last (lowest-resolution) conv layer — a higher-resolution target layer (an
earlier block) or a higher input resolution would be needed to tell the
difference and to get clinically legible localization. As-is, these heatmaps
support only a coarse sanity check, not fine-grained lesion attribution.

## Regulatory context (informational)

For software in a medical device, FDA design controls under **21 CFR Part 820.30**
require, among other things, design verification and **design validation** —
objective evidence that the device meets user needs and intended uses, and that
the design outputs meet the design inputs.

The artifacts in this repo map onto that framework as follows:

| Design-control element (21 CFR 820.30) | Artifact here |
|---|---|
| Design inputs | Intended use: 5-class DR severity from fundus photos; performance requirement stated as a QWK threshold |
| Design outputs | Trained model weights + the deterministic `predict.py` inference path |
| Verification (outputs meet inputs) | Held-out QWK, per-class accuracy, confusion matrix on a data split not seen in training |
| Validation (meets user needs / intended use) | Evaluation on data representative of the deployment population; Grad-CAM review that the model attends to pathology, not artifacts |
| Design history file | Version-controlled code, config, metrics, and the augmentation/label-handling decisions recorded above |

Grad-CAM specifically supports the validation argument: it produces the
per-case interpretability evidence a reviewer expects to see when assessing
whether an image model's decisions are clinically grounded. A real submission
would also require a locked dataset with documented provenance and grader
agreement, prospective or independent test data, subgroup performance analysis,
human-factors evaluation, and a full risk-management file (ISO 14971). None of
that is included here.

## Limitations

- **Grad-CAM heatmaps are coarse (10×10 upsampled 30×) and, on manual review,
  often concentrate near the fundus border rather than clearly on vessels or
  lesions.** See the caveat under Explainability. This weakens the
  interpretability claim the Regulatory context section makes — as-is, the
  heatmaps are a sanity check, not lesion-level evidence.
- **Merged, not curated, dataset.** Training data is APTOS 2019 combined with
  EyePACS, not a single source with one labeling protocol. The two sets were
  graded independently and were captured on different camera hardware and
  patient populations; merging them is a domain-shift risk that hasn't been
  measured or corrected for.
- **No external validation set.** Every number in the Results table comes from
  a held-out split of the *same* merged APTOS+EyePACS pool (de-leaked at the
  source-image level, see Dataset above). None of it is an independent dataset
  or a different clinic/camera population — the standard bar for a
  generalization claim in DR grading.
- **No subgroup analysis.** Performance has not been broken out by camera
  type, patient demographics, or image quality. The reported metrics are
  pooled averages and could mask large disparities across subgroups.
- **Grade 1 (Mild DR) is the model's weak point, and it did not improve with
  more training capacity.** Per-class F1 for grade 1 is 0.09 (baseline,
  frozen backbone) and 0.04 (fine-tuned, full backbone) vs. 0.60–0.92 for
  every other grade; fine-tuning actually dropped grade-1 recall from 0.08 to
  0.03 — see Results. The grade 0/1 boundary is a known-hard case in DR
  grading generally (subtle microaneurysms, high inter-rater disagreement in
  the literature), but that a full-backbone fine-tune made it *worse* points
  to something more specific: likely label noise at this boundary in the
  merged dataset, or the class-weighted loss and per-class support (only 399
  val / 398 test grade-1 images) not being enough to counter how visually
  similar grade 0/1 are. Not resolved in this project.
- **Nonstandard split methodology.** `train` contains multiple offline-augmented
  copies per source image, and `data.py` de-leaks val/test against `train` and
  restricts them to original (un-augmented) images to keep metrics honest (see
  Dataset → Split integrity). This mitigates the main risk of augmented
  duplicates leaking across splits, but it's still a different, less-standard
  setup than a benchmark curated from the start with clean, fixed splits.
- **Not a cleared or approved medical device** — see the disclaimer at the top
  of this README and the "None of that is included here" caveat in Regulatory
  context above.
- **Single-model, single-run results.** No ensembling and no repeated-seed
  variance estimate. The reported QWK and other point estimates come from one
  training run each (baseline complete; a full fine-tune with discriminative
  learning rates was in progress at time of writing) and should be read as
  single samples, not stable means.

## Usage

```bash
pip install -r requirements.txt

# full run: train 20 epochs, then evaluate best.pt on the test split (resumable)
powershell -ExecutionPolicy Bypass -File run_training.ps1

# or directly
python train.py --data augmented_resized_V2 --epochs 20 --batch-size 16 [--resume auto]
python evaluate.py --split test --ckpt checkpoints/best.pt

# grade one image: prints grade + confidence, writes <image>_gradcam.png
python predict.py path/to/fundus.jpg --ckpt checkpoints/best.pt

# self-check on the Grad-CAM math (no dataset needed)
python test_gradcam.py
```

## Files

| File | Purpose |
|---|---|
| `data.py` | Albumentations transforms, dataset, de-leaking + originals-only eval loaders, class weights |
| `model.py` | EfficientNet-B3 builder + layer-freezing + Grad-CAM target layer |
| `gradcam.py` | `GradCAM`, `overlay_gradcam`, `batch_overlay` |
| `train.py` | Seeded training loop, AMP, cosine LR, per-epoch QWK/CM, last.pt + best.pt, `--resume` |
| `evaluate.py` | QWK / confusion matrix / per-class + referable-DR sens-spec on test or val; CM png |
| `predict.py` | Single-image inference → grade + confidence + heatmap PNG |
| `run_training.ps1` | One-shot train → evaluate, logs to `logs/` |
| `test_gradcam.py` | Runnable check of the Grad-CAM computation |
