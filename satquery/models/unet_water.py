from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "water_unet.pt"
_MODEL = None


def _torch():
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError:
        return None, None


def build_unet():
    torch, nn = _torch()
    if torch is None:
        return None

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        def forward(self, x): return self.net(x)

    class WaterUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = Block(3, 16), Block(16, 32), Block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.mid = Block(64, 128)
            self.u3, self.d3 = nn.ConvTranspose2d(128, 64, 2, 2), Block(128, 64)
            self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), Block(64, 32)
            self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), Block(32, 16)
            self.out = nn.Conv2d(16, 1, 1)
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); x = self.mid(self.pool(e3))
            x = self.d3(torch.cat([self.u3(x), e3], 1)); x = self.d2(torch.cat([self.u2(x), e2], 1)); x = self.d1(torch.cat([self.u1(x), e1], 1))
            return self.out(x)
    return WaterUNet()


def load_unet():
    global _MODEL
    torch, _ = _torch()
    if torch is None or not MODEL_PATH.exists():
        return None
    if _MODEL is None:
        bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = build_unet()
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        _MODEL = (model, bundle)
    return _MODEL


def predict_water_unet(image: Image.Image):
    loaded = load_unet()
    if loaded is None:
        return None
    torch, _ = _torch(); model, bundle = loaded
    size = int(bundle.get("input_size", 256))
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.no_grad(): probability = torch.sigmoid(model(x))[0, 0].numpy()
    probability = np.asarray(Image.fromarray(probability.astype(np.float32), mode="F").resize(image.size, Image.Resampling.BILINEAR))
    threshold = float(bundle.get("threshold", 0.5))
    return probability >= threshold, probability, bundle
