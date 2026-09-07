from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    label: str
    score: float
    factors: dict[str, float]


def combine(model_score: float, input_quality: float = 1.0, alignment_quality: float = 1.0) -> Confidence:
    factors = {
        "model_signal": round(max(0.0, min(1.0, model_score)), 3),
        "input_quality": round(max(0.0, min(1.0, input_quality)), 3),
        "alignment_quality": round(max(0.0, min(1.0, alignment_quality)), 3),
    }
    score = factors["model_signal"] * factors["input_quality"] * factors["alignment_quality"]
    label = "high" if score >= 0.72 else "medium" if score >= 0.45 else "low"
    return Confidence(label, round(score, 3), factors)

