"""One runnable check that the Grad-CAM math holds: run `python test_gradcam.py`.

Uses a tiny CNN so it needs no dataset or pretrained download.
"""
import numpy as np
import torch
import torch.nn as nn

from gradcam import GradCAM


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


def main():
    torch.manual_seed(0)
    net = TinyNet().eval()
    cam_fn = GradCAM(net, net.features[2])  # last conv
    x = torch.randn(1, 3, 32, 32)

    cam, cls = cam_fn(x, class_idx=3)
    assert cam.shape == (32, 32), cam.shape
    assert cls == 3
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6, (cam.min(), cam.max())
    assert np.isclose(cam.max(), 1.0, atol=1e-3), cam.max()  # normalized to [0,1]

    cam2, cls2 = cam_fn(x)  # argmax path
    assert cam2.shape == (32, 32) and 0 <= cls2 < 5
    print("ok")


if __name__ == "__main__":
    main()
