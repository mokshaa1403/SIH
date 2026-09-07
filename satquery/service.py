from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import time

from satquery.imaging import decode_data_url, identify_image_type, validate_count
from satquery.models import analyze_change, analyze_fusion, analyze_grounding, analyze_vqa
from satquery.router import route


MODES = {"single_optical", "single_sar", "temporal_pair", "optical_sar_pair"}


def analyze(payload: dict) -> dict:
    started = time.perf_counter()
    mode = payload.get("mode", "single_optical")
    question = str(payload.get("question", "")).strip()
    if mode not in MODES:
        raise ValueError(f"Unsupported input mode: {mode}")
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("The images field must be a list.")
    images = [decode_data_url(item.get("data", "")) for item in raw_images if isinstance(item, dict)]
    validate_count(mode, images)
    image_types = [identify_image_type(image) for image in images]
    selected = route(question, mode)

    trace = [
        {"step": "validation", "detail": f"Accepted {len(images)} image(s) in {mode} mode."},
        {"step": "image identification", "detail": "; ".join(f"Image {i + 1}: {info['label']} ({info['confidence'] * 100:.0f}% heuristic confidence)" for i, info in enumerate(image_types))},
        {"step": "privacy", "detail": "Processed the uploaded imagery on the local SatQuery server; no external catalogue or API was called."},
        {"step": "routing", "detail": selected.reason},
    ]
    if selected.task == "unsupported":
        result = {
            "answer": "This question is outside the supported prototype workflows. Try asking about visible land cover, locating a supported class, change over time, or optical–SAR evidence.",
            "metrics": {},
            "confidence": None,
            "warnings": ["No model was run."],
        }
    elif selected.task == "vqa":
        result = analyze_vqa(images[0], selected.target)
    elif selected.task == "grounding":
        if mode != "single_optical":
            raise ValueError("The grounding baseline currently requires single optical mode.")
        result = analyze_grounding(images[0], selected.target)
    elif selected.task == "change_detection":
        if len(images) != 2:
            raise ValueError("Change detection requires a before and an after image.")
        result = analyze_change(images[0], images[1])
    else:
        if mode != "optical_sar_pair":
            raise ValueError("Optical–SAR fusion requires optical–SAR pair mode.")
        result = analyze_fusion(images[0], images[1], selected.target)

    if result.get("confidence") is not None:
        result["confidence"] = asdict(result["confidence"])
    trace.append({"step": "analysis", "detail": f"Executed {selected.task.replace('_', ' ')} specialist."})
    trace.append({"step": "evidence", "detail": "Returned structured metrics and visual evidence where supported."})
    trace.append({"step": "review", "detail": "The user must verify the evidence before operational use."})
    return {
        "request_id": datetime.now(timezone.utc).strftime("sq-%Y%m%d%H%M%S%f"),
        "task": selected.task,
        "target": selected.target,
        "mode": mode,
        "input_images": image_types,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "trace": trace,
        **result,
    }
