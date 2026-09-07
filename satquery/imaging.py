from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable

import numpy as np
from PIL import Image

MAX_SIDE = 1280


def decode_data_url(data: str) -> Image.Image:
    if not isinstance(data, str) or "," not in data:
        raise ValueError("Image data must be a browser data URL.")
    header, encoded = data.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Only image uploads are supported.")
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(BytesIO(raw)).convert("RGB")
        image.load()
    except Exception as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    if min(image.size) < 32:
        raise ValueError("Images must be at least 32 by 32 pixels.")
    return image


def as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_like(image: Image.Image, reference: Image.Image) -> Image.Image:
    return image.resize(reference.size, Image.Resampling.BILINEAR)


def connected_cleanup(mask: np.ndarray, minimum_fraction: float = 0.001) -> np.ndarray:
    # Lightweight neighborhood consensus that removes isolated pixel noise.
    m = mask.astype(np.uint8)
    padded = np.pad(m, 1)
    votes = sum(padded[y:y + m.shape[0], x:x + m.shape[1]] for y in range(3) for x in range(3))
    cleaned = votes >= 5
    if cleaned.mean() < minimum_fraction:
        return mask
    return cleaned


def identify_image_type(image: Image.Image) -> dict:
    """Distinguish colour-coded RGB composites from ordinary natural-colour images.

    This is a transparent heuristic, not sensor metadata inference. A PNG/JPEG has
    only display channels, so the original satellite bands cannot be recovered.
    """
    rgb = as_array(image)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = maximum - minimum
    dominant = (maximum > 0.35) & (saturation > 0.38)
    primary_gap = np.sort(rgb, axis=2)[..., 2] - np.sort(rgb, axis=2)[..., 1]
    primary_fraction = float((dominant & (primary_gap > 0.18)).mean())
    highly_saturated_fraction = float((saturation > 0.48).mean())
    score = min(1.0, 0.65 * primary_fraction / 0.22 + 0.35 * highly_saturated_fraction / 0.30)
    kind = "rgb_composite" if score >= 0.55 else "normal_optical"
    return {
        "type": kind,
        "label": "RGB / false-colour composite" if kind == "rgb_composite" else "normal natural-colour image",
        "confidence": round(max(score, 1.0 - score), 3),
        "composite_score": round(score, 3),
        "method": "display-channel colour distribution heuristic",
    }


def spectral_masks(image: Image.Image, image_type: str | None = None) -> dict[str, np.ndarray]:
    rgb = as_array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    # Natural-colour water may be blue/cyan, dark, or mildly turbid. False-colour
    # composites commonly render it as a dark low-reflectance or blue region.
    blue_or_cyan = (b > r * 1.06) & (b >= g * 0.90) & (brightness < 0.82)
    dark_low_reflectance = (brightness < 0.24) & (saturation < 0.42)
    neighbour_gradient = (
        np.abs(brightness - np.roll(brightness, 1, 0))
        + np.abs(brightness - np.roll(brightness, 1, 1))
    ) / 2
    smooth_dark = (brightness < 0.38) & (neighbour_gradient < 0.055) & (g <= r * 1.18)
    if image_type == "rgb_composite":
        water = blue_or_cyan | dark_low_reflectance | smooth_dark
    else:
        water = blue_or_cyan | dark_low_reflectance | (smooth_dark & (b >= r * 0.82))
    vegetation = (g > r * 1.10) & (g > b * 1.04) & (g > 0.22)
    built_up = (saturation < 0.14) & (brightness > 0.38) & (brightness < 0.82)
    bare_land = (r > b * 1.08) & (r >= g * 0.92) & (brightness > 0.28) & ~vegetation
    red_region = (r > g * 1.18) & (r > b * 1.18) & (r > 0.25)
    green_region = (g > r * 1.18) & (g > b * 1.12) & (g > 0.25)
    blue_region = (b > r * 1.18) & (b > g * 1.12) & (b > 0.25)

    return {name: connected_cleanup(mask) for name, mask in {
        "water": water,
        "vegetation": vegetation,
        "built_up": built_up,
        "bare_land": bare_land,
        "red_region": red_region,
        "green_region": green_region,
        "blue_region": blue_region,
    }.items()}


COLORS = {
    "water": (33, 150, 243),
    "vegetation": (76, 175, 80),
    "built_up": (255, 152, 0),
    "bare_land": (156, 111, 72),
    "change": (239, 68, 68),
    "fusion": (168, 85, 247),
    "red_region": (255, 235, 59),
    "green_region": (255, 235, 59),
    "blue_region": (255, 235, 59),
}


def overlay(image: Image.Image, mask: np.ndarray, label: str, alpha: float = 0.48) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.array(COLORS[label], dtype=np.float32)
    result = base.copy()
    result[mask] = result[mask] * (1 - alpha) + color * alpha
    # Add a bright one-pixel boundary.
    eroded = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        eroded[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
            & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
    boundary = mask & ~eroded
    result[boundary] = color
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def required_image_count(mode: str) -> int:
    return 2 if mode in {"temporal_pair", "optical_sar_pair"} else 1


def validate_count(mode: str, images: Iterable[Image.Image]) -> None:
    count = len(list(images))
    expected = required_image_count(mode)
    if count != expected:
        raise ValueError(f"Mode '{mode}' requires exactly {expected} image(s); received {count}.")
