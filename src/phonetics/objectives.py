"""Objective identifiers and arbitrary finite acoustic loss weights."""
from __future__ import annotations

import math
import re
from numbers import Real

BASE_MODEL_IDS = ("syllable", "components", "acoustic")
COMBINED_PREFIX = "components-acoustic-lambda-"


def combined_model_id(weight: Real) -> str:
    """Encode a finite coefficient without truncating its float precision."""
    if isinstance(weight, bool) or not isinstance(weight, Real):
        raise ValueError("lambda must be a finite real number")
    try:
        value = float(weight)
    except (OverflowError, ValueError) as exc:
        raise ValueError("lambda must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError("lambda must be a finite real number")
    token = str(int(value)) if value.is_integer() and abs(value) < 1e16 else repr(value).replace("e+", "e")
    return COMBINED_PREFIX + token


def parse_model_id(model_id: str) -> tuple[str, float | None]:
    """Return the objective family and the combined acoustic coefficient."""
    if isinstance(model_id, str) and model_id in BASE_MODEL_IDS:
        return model_id, None
    if not isinstance(model_id, str) or not model_id.startswith(COMBINED_PREFIX):
        raise ValueError("unknown objective model_id")
    token = model_id[len(COMBINED_PREFIX):]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        raise ValueError("combined model_id must contain a finite numeric lambda")
    weight = float(token)
    if not math.isfinite(weight):
        raise ValueError("lambda must be a finite real number")
    return "components-acoustic", weight


def canonical_model_id(model_id: str) -> str:
    family, weight = parse_model_id(model_id)
    return combined_model_id(weight) if family == "components-acoustic" else family
