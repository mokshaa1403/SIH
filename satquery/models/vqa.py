from PIL import Image

from satquery.confidence import combine
from satquery.imaging import spectral_masks
from satquery.models.scene import predict_scene


LABELS = {
    "water": "water", "vegetation": "vegetation", "built_up": "built-up land", "bare_land": "bare land",
    "red_region": "red-coloured region", "green_region": "green-coloured region", "blue_region": "blue-coloured region",
}


def analyze_vqa(image: Image.Image, target: str | None) -> dict:
    masks = spectral_masks(image)
    fractions = {name: round(float(mask.mean()) * 100, 2) for name, mask in masks.items()}
    scene = predict_scene(image)
    if target in masks:
        coverage = fractions[target]
        present = coverage >= 1.0
        answer = f"{LABELS[target].capitalize()} is {'visible' if present else 'not clearly visible'} in the image."
        signal = min(0.92, 0.48 + abs(coverage - 1.0) / 30)
    else:
        ranked = sorted(fractions.items(), key=lambda item: item[1], reverse=True)
        visible = [LABELS[name] for name, value in ranked if value >= 1.0][:3]
        spectral_text = ", ".join(visible) if visible else "no supported land-cover class with enough evidence"
        if scene:
            answer = f"The local scene classifier predicts '{scene['label']}'. The separate spectral baseline detects {spectral_text}."
            signal = max(0.35, min(0.86, scene["confidence"] * 0.82))
        else:
            answer = "The baseline detects " + spectral_text + "."
            signal = 0.64 if visible else 0.35
    confidence = combine(signal)
    return {
        "answer": answer,
        "metrics": {"estimated_class_coverage_percent": fractions, "scene_prediction": scene or "model_not_trained"},
        "confidence": confidence,
        "warnings": ["This answer uses baseline features and is not a trained satellite VQA conclusion."] + (["Mars and Moon predictions are outside Earth-observation scope."] if scene and scene["label"] in {"mars", "moon"} else []),
    }
