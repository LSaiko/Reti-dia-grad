"""Grade a single fundus image: prints grade + confidence, writes a Grad-CAM heatmap PNG."""
import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from data import IMG_SIZE
from gradcam import GradCAM, overlay_gradcam
from model import build_model, target_layer

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=5, pretrained=False, freeze=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt.get("img_size", IMG_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", help="heatmap PNG path (default: <image>_gradcam.png)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, img_size = load_model(args.ckpt, device)

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
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
    main()
