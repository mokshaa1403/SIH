from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "model_registry.json"
SCENE_MODEL = ROOT / "artifacts" / "scene_classifier.npz"
WATER_MODEL = ROOT / "artifacts" / "water_segmenter.pkl"


def model_card() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    scene = {"available": SCENE_MODEL.exists()}
    if SCENE_MODEL.exists():
        data = np.load(SCENE_MODEL, allow_pickle=False)
        scene.update({
            "training_images": int(data["training_images"]) if "training_images" in data else int(len(data["labels"])),
            "classes": [str(value) for value in (data["classes"] if "classes" in data else np.unique(data["labels"]))],
            "training_protocol": str(data["training_protocol"]) if "training_protocol" in data else "legacy_or_unknown",
            "internal_test_accuracy": None if "test_accuracy" not in data or np.isnan(float(data["test_accuracy"])) else round(float(data["test_accuracy"]), 4),
        })
    water = {"available": WATER_MODEL.exists()}
    if WATER_MODEL.exists():
        with WATER_MODEL.open("rb") as handle:
            water.update(pickle.load(handle).get("metadata", {}))
    return {
        "deployment": "local_only_by_default",
        "data_upload": "images remain on this local server",
        "scene_classifier": scene,
        "water_segmenter": water,
        "registry": registry,
        "limitations": [
            "Normal-image water grounding uses supervised image-mask training; RGB composites still use a separate rule until labelled composite masks are supplied.",
            "Change detection and optical SAR fusion remain transparent baselines until paired task labels are supplied.",
            "Operational use requires sensor specific evaluation and human review.",
        ],
    }
