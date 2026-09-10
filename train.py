"""Train the EfficientNet-B3 DR grader. Logs QWK (the APTOS metric) each epoch.

Built for an unattended run: seeded, cosine LR schedule, writes checkpoints/last.pt every
epoch and checkpoints/best.pt on val-QWK improvement. Re-run with --resume auto to continue.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from data import IMG_SIZE, class_weights, make_loaders
from model import build_model

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick fast kernels


def evaluate(model, loader, device, amp=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        for x, y in loader:
            out = model(x.to(device))
            preds.append(out.argmax(1).cpu().numpy())
            targets.append(y.numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    acc = float((preds == targets).mean())
    labels = list(range(5))
    cm = confusion_matrix(targets, preds, labels=labels)
    report = classification_report(targets, preds, labels=labels,
                                   target_names=GRADE_NAMES, zero_division=0, digits=3)
    return qwk, acc, cm, report


def print_eval(cm, report):
    print("  confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{i:>6}" for i in range(5)))
    for i, row in enumerate(cm):
        print(f"    {i} | " + " ".join(f"{v:>6}" for v in row))
    print("  per-class metrics:")
    for line in report.splitlines():
        print("    " + line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="augmented_resized_V2", help="root with train/ val/ subdirs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-freeze", action="store_true", help="train the whole backbone")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (CUDA only)")
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for <out>/last.pt")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, _, train_ld, val_ld = make_loaders(
        args.data, args.img_size, args.batch_size, args.workers)
    print(f"train {len(train_ds)}  val {len(val_ld.dataset)}  device {device}  amp {not args.no_amp}")

    model = build_model(num_classes=5, freeze=not args.no_freeze).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_qwk = 1, -1.0
    resume_path = out / "last.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_qwk = ck["epoch"] + 1, ck["best_qwk"]
        print(f"resumed from {resume_path} at epoch {start_epoch} (best QWK {best_qwk:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for i, (x, y) in enumerate(train_ld):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if (i + 1) % 50 == 0:
                seen = (i + 1) * train_ld.batch_size
                print(f"  epoch {epoch} batch {i + 1}/{len(train_ld)} "
                      f"loss {running / (i + 1):.4f}  {seen / (time.time() - t0):.0f} img/s", flush=True)
        scheduler.step()

        train_secs = time.time() - t0
        qwk, acc, cm, report = evaluate(model, val_ld, device, amp)
        print(f"epoch {epoch}/{args.epochs}: train_loss {running / len(train_ld):.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  "
              f"train {train_secs:.0f}s ({len(train_ld) * train_ld.batch_size / train_secs:.0f} img/s)  "
              f"val_acc {acc:.4f}  val_QWK {qwk:.4f}", flush=True)
        print_eval(cm, report)

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_qwk": max(best_qwk, qwk),
                "classes": train_ds.classes, "img_size": args.img_size, "qwk": qwk}
        torch.save(ckpt, out / "last.pt")
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(ckpt, out / "best.pt")
            print(f"  saved best.pt (QWK {qwk:.4f})", flush=True)

    print(f"done. best val QWK {best_qwk:.4f}")


if __name__ == "__main__":
    main()
