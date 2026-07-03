from __future__ import annotations

from typing import Dict


def edge_confidence(
    continuity: float,
    proximity: float,
    alignment: float,
    node_support: float,
) -> float:
    return (
        0.45 * continuity
        + 0.25 * proximity
        + 0.20 * alignment
        + 0.10 * node_support
    )


def confidence_status(confidence: float) -> str:
    if confidence >= 0.65:
        return "accepted"
    if confidence >= 0.40:
        return "uncertain"
    return "rejected"


def parse_image_stub(_image_path: str) -> Dict[str, object]:
    raise NotImplementedError(
        "Image parsing is dataset-dependent. This reproduction package expects "
        "diagram-derived graphs as JSON. Connect OCR/OpenCV extraction here if "
        "raw images are available."
    )

