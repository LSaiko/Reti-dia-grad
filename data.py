"""Dataset, Albumentations pipeline, class weights.

Expects an ImageFolder layout: <root>/{train,val,test}/{0,1,2,3,4}/*.jpg
(as produced by APTOS 2019 after grouping images by grade).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils import class_weight

# EfficientNet-B3 native resolution; ImageNet normalization matches timm pretrained weights.
IMG_SIZE = 300


def build_transforms(img_size=IMG_SIZE, train=True):
    if not train:
        return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(p=0.3),  # ponytail: library defaults for hole count/size; widen via kwargs if the model ignores lesions
        A.Normalize(),
        ToTensorV2(),
    ])


def base_id(path):
    """Source fundus-image id: leading token before the first '-' (e.g. '<hex>-600-FA.jpg' -> '<hex>')."""
    return Path(path).stem.split("-", 1)[0]


class AlbFolder(datasets.ImageFolder):
    """ImageFolder that feeds an Albumentations Compose instead of a torchvision transform.

    originals_only:   keep only un-augmented files (stem is exactly '<id>-600'). Use for
                      val/test so metrics aren't computed over correlated augmented copies.
    exclude_base_ids: drop files whose base_id is in this set. Use to remove train/val
                      leakage (the offline splits share ~8% of source images across splits).
    """

    def __init__(self, *args, originals_only=False, exclude_base_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        exclude = exclude_base_ids or set()
        kept = self.samples
        if originals_only:
            kept = [s for s in kept if Path(s[0]).stem.split("-", 1)[-1] == "600"]
        if exclude:
            kept = [s for s in kept if base_id(s[0]) not in exclude]
        if originals_only or exclude:
            assert kept, f"filters left 0 samples in {self.root} — wrong dataset layout?"
            self.samples = self.imgs = kept
            self.targets = [t for _, t in kept]

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        return self.transform(image=img)["image"], target


def preprocess(rgb_uint8, img_size=IMG_SIZE):
    """np.uint8 RGB HxWx3 -> normalized 1x3xHxW tensor (val pipeline)."""
    tf = build_transforms(img_size, train=False)
    return tf(image=rgb_uint8)["image"].unsqueeze(0)


def class_weights(dataset):
    labels = [y for _, y in dataset.samples]
    w = class_weight.compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(w, dtype=torch.float)


def make_loaders(root, img_size=IMG_SIZE, batch_size=16, workers=8, deleak=True):
    root = Path(root)
    train_ds = AlbFolder(root / "train", transform=build_transforms(img_size, train=True))
    train_ids = {base_id(p) for p, _ in train_ds.samples} if deleak else None
    val_ds = AlbFolder(root / "val", transform=build_transforms(img_size, train=False),
                       originals_only=True, exclude_base_ids=train_ids)
    kw = dict(num_workers=workers, pin_memory=True,
              persistent_workers=workers > 0,  # Windows spawn: don't re-import torch per epoch
              prefetch_factor=4 if workers > 0 else None)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_ds, val_ds, train_ld, val_ld


def make_eval_loader(root, split, img_size=IMG_SIZE, batch_size=16, workers=8, deleak=True):
    """Loader for val/ or test/: originals only, base-ids overlapping train/ removed."""
    root = Path(root)
    train_ids = None
    if deleak:
        tr = datasets.ImageFolder(root / "train")
        train_ids = {base_id(p) for p, _ in tr.samples}
    ds = AlbFolder(root / split, transform=build_transforms(img_size, train=False),
                   originals_only=True, exclude_base_ids=train_ids)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=True, prefetch_factor=4 if workers else None)
    return ds, ld
