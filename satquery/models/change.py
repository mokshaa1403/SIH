import numpy as np
from PIL import Image, ImageFilter

from satquery.confidence import combine
from satquery.imaging import as_array, connected_cleanup, encode_png, overlay, resize_like


def analyze_change(before: Image.Image, after: Image.Image) -> dict:
    original_size_match = before.size == after.size
    if not original_size_match:
        after = resize_like(after, before)
    a = as_array(before.filter(ImageFilter.GaussianBlur(1.2)))
    b = as_array(after.filter(ImageFilter.GaussianBlur(1.2)))
    delta = np.mean(np.abs(a - b), axis=2)
    adaptive = max(0.12, float(np.quantile(delta, 0.88)))
    mask = connected_cleanup(delta >= adaptive)
    changed = float(mask.mean()) * 100
    mean_delta = float(delta.mean())
    alignment = 1.0 if original_size_match else 0.72
    signal = min(0.91, 0.45 + mean_delta * 2.3)
    confidence = combine(signal, alignment_quality=alignment)
    answer = f"The baseline marks approximately {changed:.1f}% of the image as substantially changed."
    warnings = ["Pixel differences can represent lighting, season, clouds, or misalignment—not only physical change."]
    if not original_size_match:
        warnings.append("The second image was resized; use genuinely co-registered imagery for meaningful results.")
    return {
        "answer": answer,
        "metrics": {"changed_area_percent": round(changed, 2), "mean_normalized_difference": round(mean_delta, 4)},
        "confidence": confidence,
        "evidence_image": encode_png(overlay(after, mask, "change")),
        "warnings": warnings,
    }

