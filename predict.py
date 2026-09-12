"""Grade a single fundus image: prints grade + confidence, writes a Grad-CAM heatmap PNG."""
import argparse
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from data import IMG_SIZE
from gradcam import GradCAM, overlay_gradcam
from model import build_model, target_layer

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Three trained checkpoints exist (see README Results); this one is the default because a
# screening tool that can't see Mild DR at all (the unboosted fine-tune: grade-1 recall 0.03)
# is more dangerous than one with slightly lower overall QWK. checkpoints_ft/best.pt has the
# higher QWK (0.723 vs 0.710) if that's what a given use case actually wants instead.
DEFAULT_CKPT = "checkpoints_grade1fix/best.pt"


class PredictError(Exception):
    """A known, expected failure - printed as a one-line message, no traceback."""


def load_model(ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise PredictError(
            f"checkpoint not found: {ckpt_path}\n"
            f"  train one first (python train.py) or pass --ckpt <path>")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=5, pretrained=False, freeze=False)
    try:
        model.load_state_dict(ckpt["model"])
    except (KeyError, RuntimeError) as e:
        raise PredictError(f"checkpoint at {ckpt_path} doesn't match this model "
                           f"(architecture mismatch or not a train.py checkpoint): {e}")
    model.to(device).eval()
    return model, ckpt.get("img_size", IMG_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", help="heatmap PNG path (default: <image>_gradcam.png)")
    args = ap.parse_args()

    if not Path(args.image).exists():
        raise PredictError(f"image not found: {args.image}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, img_size = load_model(args.ckpt, device)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise PredictError(f"could not read image (corrupt file or unsupported format): {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from data import preprocess
    with torch.no_grad():
        probs = F.softmax(model(preprocess(rgb, img_size).to(device)), dim=1)[0].cpu()
    grade = int(probs.argmax())
    confidence = float(probs[grade])

    cam = GradCAM(model, target_layer(model))
    blend, _ = overlay_gradcam(args.image, model, cam, class_idx=grade, img_size=img_size)
    out_path = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + "_gradcam.png")
    cv2.imwrite(str(out_path), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

    print(f"grade      : {grade} ({GRADE_NAMES[grade]})")
    print(f"confidence : {confidence:.3f}")
    print("per-class  : " + "  ".join(f"{i}:{p:.3f}" for i, p in enumerate(probs.tolist())))
    print(f"heatmap    : {out_path}")


if __name__ == "__main__":
    try:
        main()
    except PredictError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
