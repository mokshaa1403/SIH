import numpy as np
from PIL import Image

from satquery.confidence import combine
from satquery.imaging import as_array, encode_png, overlay, resize_like, spectral_masks


def analyze_fusion(optical: Image.Image, sar: Image.Image, target: str | None) -> dict:
    original_size_match = optical.size == sar.size
    if not original_size_match:
        sar = resize_like(sar, optical)
    masks = spectral_masks(optical)
    sar_gray = as_array(sar).mean(axis=2)
    # Dark smooth SAR regions support water; bright returns support structures.
    if target == "built_up":
        optical_mask = masks["built_up"]
        sar_support = sar_gray > float(np.quantile(sar_gray, 0.68))
    else:
        target = "water"
        optical_mask = masks["water"]
        sar_support = sar_gray < float(np.quantile(sar_gray, 0.32))
    fused = (optical_mask.astype(float) * 0.62 + sar_support.astype(float) * 0.38) >= 0.52
    agreement = float((optical_mask == sar_support).mean())
    coverage = float(fused.mean()) * 100
    alignment = 1.0 if original_size_match else 0.7
    confidence = combine(0.48 + agreement * 0.42, alignment_quality=alignment)
    warnings = ["This is score-level baseline fusion, not a trained optical–SAR foundation model."]
    if not original_size_match:
        warnings.append("The SAR image was resized; operational fusion requires verified co-registration.")
    return {
        "answer": f"Optical and SAR evidence jointly indicate approximately {coverage:.1f}% potential {target.replace('_', ' ')} coverage.",
        "metrics": {
            "target": target,
            "fused_coverage_percent": round(coverage, 2),
            "optical_sar_agreement_percent": round(agreement * 100, 2),
            "fusion_weights": {"optical": 0.62, "sar": 0.38},
        },
        "confidence": confidence,
        "evidence_image": encode_png(overlay(optical, fused, "fusion")),
        "warnings": warnings,
    }

