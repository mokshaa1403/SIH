from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "scene_classifier.npz"


def extract_features(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[..., channel], bins=16, range=(0, 1), density=True)
        features.extend(hist.tolist())
        features.extend([float(rgb[..., channel].mean()), float(rgb[..., channel].std())])
    gray = np.asarray(rgb_image.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    features.extend([float(gray.mean()), float(gray.std()), float(gx), float(gy)])
    # Coarse spatial appearance helps separate planetary, water, and mountain scenes.
    thumb = np.asarray(rgb_image.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
    features.extend(thumb.tolist())
    return np.asarray(features, dtype=np.float32)


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    data = np.load(model_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def predict_scene(image: Image.Image) -> dict | None:
    model = load_model()
    if model is None:
        return None
    feature = extract_features(image)
    normalized = (feature - model["mean"]) / model["std"]
    distances = np.mean((model["features"] - normalized) ** 2, axis=1)
    k = min(5, len(distances))
    nearest = np.argpartition(distances, k - 1)[:k]
    labels = model["labels"][nearest]
    unique, counts = np.unique(labels, return_counts=True)
    winner = str(unique[np.argmax(counts)])
    agreement = float(counts.max() / k)
    nearest_distance = float(distances[nearest].min())
    confidence = max(0.0, min(1.0, agreement * (1.0 / (1.0 + nearest_distance * 0.08))))
    return {"label": winner, "confidence": round(confidence, 3), "neighbor_agreement": round(agreement, 3)}
