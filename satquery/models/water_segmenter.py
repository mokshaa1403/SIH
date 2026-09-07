from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label, uniform_filter

from satquery.imaging import connected_cleanup

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "water_segmenter.pkl"
_MODEL = None


def pixel_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum, minimum = rgb.max(2), rgb.min(2)
    brightness = rgb.mean(2)
    saturation = maximum - minimum
    grad = (np.abs(brightness - np.roll(brightness, 1, 0)) + np.abs(brightness - np.roll(brightness, 1, 1))) / 2
    mean3 = uniform_filter(brightness, size=3, mode="reflect")
    mean9 = uniform_filter(brightness, size=9, mode="reflect")
    mean21 = uniform_filter(brightness, size=21, mode="reflect")
    std5 = np.sqrt(np.maximum(0, uniform_filter(brightness * brightness, size=5, mode="reflect") - uniform_filter(brightness, size=5, mode="reflect") ** 2))
    std15 = np.sqrt(np.maximum(0, uniform_filter(brightness * brightness, size=15, mode="reflect") - uniform_filter(brightness, size=15, mode="reflect") ** 2))
    local_r = uniform_filter(r, size=11, mode="reflect")
    local_g = uniform_filter(g, size=11, mode="reflect")
    local_b = uniform_filter(b, size=11, mode="reflect")
    features = np.stack([r, g, b, brightness, saturation, maximum, minimum, b-r, b-g, g-r,
                         b/(r+g+b+0.03), g/(r+g+b+0.03), grad, mean3, mean9, mean21,
                         std5, std15, local_r, local_g, local_b], axis=2)
    return features.reshape(-1, features.shape[2])


def refine_water_mask(mask: np.ndarray, probability: np.ndarray, image: Image.Image) -> np.ndarray:
    """Suppress fragmented shadow predictions while retaining coherent water regions."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    brightness = rgb.mean(2)
    texture = np.sqrt(np.maximum(0, uniform_filter(brightness * brightness, 9) - uniform_filter(brightness, 9) ** 2))
    green_land = (rgb[..., 1] > rgb[..., 2] * 1.08) & (rgb[..., 1] > rgb[..., 0] * 1.06) & (texture > 0.035)
    textured_shadow = (brightness < 0.28) & (texture > 0.075)
    confident = probability >= 0.86
    refined = mask & (~green_land | confident) & (~textured_shadow | confident)
    refined = binary_closing(refined, iterations=1)
    components, count = label(refined)
    if count:
        sizes = np.bincount(components.ravel())
        minimum = max(20, int(refined.size * 0.00035))
        keep = sizes >= minimum
        keep[0] = False
        refined = keep[components]
    return binary_fill_holes(refined).astype(bool)


def load_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as handle:
            _MODEL = pickle.load(handle)
    return _MODEL


def predict_water(image: Image.Image, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    bundle = load_model()
    if bundle is None:
        return None
    probability = bundle["model"].predict_proba(pixel_features(image))[:, 1].reshape(image.height, image.width)
    selected_threshold = float(bundle.get("metadata", {}).get("threshold", 0.58) if threshold is None else threshold)
    mask = connected_cleanup(probability >= selected_threshold)
    return refine_water_mask(mask, probability, image), probability
