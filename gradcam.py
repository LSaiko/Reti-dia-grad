"""Grad-CAM for the DR grader, plus overlay helpers."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 300
_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _m, _i, output):
        self.activations = output.detach()

    def _save_gradients(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, input_tensor, class_idx=None):
        """input_tensor: 1x3xHxW. Returns (cam HxW float in [0,1], class_idx used)."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # global-average-pool the gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_gradcam(image_path, model, gradcam, class_idx=None, img_size=IMG_SIZE, alpha=0.4):
    """Load a fundus photo, run Grad-CAM, return (blended RGB uint8 at native size, class_idx)."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    from data import preprocess  # deferred so GradCAM itself needs no albumentations
    device = next(model.parameters()).device
    x = preprocess(rgb, img_size).to(device)
    cam, used = gradcam(x, class_idx)

    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = np.uint8((1 - alpha) * rgb + alpha * heat)
    return blend, used


def batch_overlay(folder, model, gradcam, out_dir="results", class_idx=None, img_size=IMG_SIZE, alpha=0.4):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in _EXTS:
            continue
        blend, _ = overlay_gradcam(p, model, gradcam, class_idx, img_size, alpha)
        cv2.imwrite(str(out / f"{p.stem}_gradcam.png"), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
        n += 1
    print(f"wrote {n} overlays to {out}/")
