"""Objective identifiers and arbitrary finite acoustic loss weights."""
from __future__ import annotations

import math
import re
from numbers import Real

BASE_MODEL_IDS = ("syllable", "components", "acoustic")
COMBINED_PREFIX = "components-acoustic-lambda-"
SYLLABLE_COMBINED_PREFIX = "syllable-acoustic-lambda-"


def combined_model_id(weight: Real, *, prefix: str = COMBINED_PREFIX) -> str:
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
    return prefix + token


def parse_model_id(model_id: str) -> tuple[str, float | None]:
    """Return the objective family and the combined acoustic coefficient."""
    if isinstance(model_id, str) and model_id in BASE_MODEL_IDS:
        return model_id, None
    prefix = next((prefix for prefix in (COMBINED_PREFIX, SYLLABLE_COMBINED_PREFIX)
                   if isinstance(model_id, str) and model_id.startswith(prefix)), None)
    if prefix is None:
        raise ValueError("unknown objective model_id")
    token = model_id[len(prefix):]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        raise ValueError("combined model_id must contain a finite numeric lambda")
    weight = float(token)
    if not math.isfinite(weight):
        raise ValueError("lambda must be a finite real number")
    return prefix.removesuffix("-lambda-"), weight


def canonical_model_id(model_id: str) -> str:
    family, weight = parse_model_id(model_id)
    return family if weight is None else combined_model_id(weight, prefix=f"{family}-lambda-")


def is_extension_model(model_id: str) -> bool:
    """Check whether an objective can reuse an existing experiment bundle."""
    return (isinstance(model_id, str) and model_id.startswith(SYLLABLE_COMBINED_PREFIX)
            and canonical_model_id(model_id) == model_id)
