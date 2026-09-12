"""Evaluate a checkpoint on the test (or val) split: QWK, accuracy, confusion matrix,
per-class metrics, and referable-DR (grades 2-4) sensitivity/specificity.
Saves results/confusion_matrix_<split>.png.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from data import IMG_SIZE, make_eval_loader
from model import build_model
from predict import DEFAULT_CKPT
from train import GRADE_NAMES, print_eval, evaluate as _evaluate


def save_cm_png(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(5), yticks=range(5), xlabel="predicted", ylabel="true",
           xticklabels=range(5), yticklabels=range(5))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="augmented_resized_V2")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(num_classes=5, pretrained=False, freeze=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    ds, ld = make_eval_loader(args.data, args.split, ck.get("img_size", IMG_SIZE),
                              workers=args.workers)
    print(f"{args.split}: {len(ds)} images")

    qwk, acc, cm, report = _evaluate(model, ld, device, amp=(device == "cuda"))
    print(f"\nQWK {qwk:.4f}   accuracy {acc:.4f}\n")
    print_eval(cm, report)

    # referable DR = grade >= 2 (needs ophthalmology referral)
    tp = cm[2:, 2:].sum(); fn = cm[2:, :2].sum()
    fp = cm[:2, 2:].sum(); tn = cm[:2, :2].sum()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    print(f"\nreferable-DR (grade>=2)  sensitivity {sens:.3f}  specificity {spec:.3f}")

    save_cm_png(cm, f"results/confusion_matrix_{args.split}.png")


if __name__ == "__main__":
    main()
