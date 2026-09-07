from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Route:
    task: str
    target: str | None
    reason: str


TARGETS = {
    "water": ("water", "river", "lake", "reservoir", "flood"),
    "vegetation": ("vegetation", "forest", "crop", "field", "green"),
    "built_up": ("building", "built", "urban", "road", "construction"),
    "bare_land": ("bare", "soil", "sand", "barren"),
    "red_region": ("red",),
    "green_region": ("green",),
    "blue_region": ("blue",),
}


def _target(question: str) -> str | None:
    for colour in ("red", "green", "blue"):
        if re.search(rf"\b{colour}\s+(pixel|pixels|region|regions|colour|color|area|areas)\b", question):
            return f"{colour}_region"
    for target, words in TARGETS.items():
        if any(re.search(rf"\b{re.escape(word)}\w*\b", question) for word in words):
            return target
    return None


def route(question: str, mode: str) -> Route:
    q = " ".join(question.lower().split())
    if not q:
        raise ValueError("Please enter a question.")

    target = _target(q)
    if mode == "temporal_pair" or any(word in q for word in ("changed", "change", "between", "before", "after")):
        return Route("change_detection", target, "The request compares observations over time.")
    if mode == "optical_sar_pair" or ("optical" in q and ("sar" in q or "radar" in q)):
        return Route("optical_sar_fusion", target, "The request requires optical and SAR evidence.")
    if any(word in q for word in ("highlight", "locate", "where", "segment", "mark", "show me")) and target:
        return Route("grounding", target, f"The request asks to locate {target.replace('_', ' ')}.")
    if any(word in q for word in ("what", "is there", "are there", "describe", "visible", "present", "how much")):
        return Route("vqa", target, "The request asks a question about one image.")
    return Route("unsupported", target, "The question does not match a supported prototype workflow.")
