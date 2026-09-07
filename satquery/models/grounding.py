from PIL import Image

from satquery.confidence import combine
from satquery.imaging import encode_png, identify_image_type, overlay, spectral_masks
from satquery.models.water_segmenter import predict_water


LABELS = {
    "water": "water", "vegetation": "vegetation", "built_up": "built-up land", "bare_land": "bare land",
    "red_region": "red-coloured", "green_region": "green-coloured", "blue_region": "blue-coloured",
}


def analyze_grounding(image: Image.Image, target: str | None) -> dict:
    if target not in LABELS:
        raise ValueError("Grounding supports water, vegetation, built-up land, bare land, and red, green, or blue regions.")
    image_info = identify_image_type(image)
    trained = predict_water(image) if target == "water" and image_info["type"] == "normal_optical" else None
    if trained is not None:
        mask, probability = trained
        method = "supervised water segmenter"
        mean_probability = float(probability[mask].mean()) if mask.any() else float(probability.mean())
    else:
        mask = spectral_masks(image, image_info["type"])[target]
        method = "RGB composite rule" if image_info["type"] == "rgb_composite" else "colour-rule fallback"
        mean_probability = None
    coverage = float(mask.mean()) * 100
    signal = min(0.9, 0.42 + coverage / 35) if coverage >= 0.5 else 0.25
    confidence = combine(signal)
    answer = (
        f"Highlighted the estimated {LABELS[target]} region, covering approximately {coverage:.1f}% of the image."
        if coverage >= 0.5 else f"No reliable {LABELS[target]} region was found by the baseline."
    )
    return {
        "answer": answer,
        "metrics": {"image_type": image_info["label"], "image_type_confidence": image_info["confidence"], "analysis_method": method, "mean_water_probability": round(mean_probability, 3) if mean_probability is not None else "not available", "estimated_coverage_percent": round(coverage, 2), "target": target},
        "confidence": confidence,
        "evidence_image": encode_png(overlay(image, mask, target)),
        "warnings": ["Normal imagery uses the supervised model. RGB composites use a separate rule until labelled composite masks are supplied. Validate the overlay before drawing conclusions."],
    }
