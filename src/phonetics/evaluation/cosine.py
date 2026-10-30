"""Numerically stable cosine calculations for raw contextual vectors."""

from __future__ import annotations

import numpy as np


def stable_cosine(left, right, *, left_id: str = "left", right_id: str = "right") -> float:
    """Compute float64 cosine after scaling each vector to avoid norm overflow."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"cosine vectors must have the same one-dimensional shape: {left_id}, {right_id}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"cosine vectors contain non-finite values: {left_id}, {right_id}")
    scale_a = float(np.max(np.abs(a))) if a.size else 0.0
    scale_b = float(np.max(np.abs(b))) if b.size else 0.0
    if scale_a == 0.0 or scale_b == 0.0:
        bad = left_id if scale_a == 0.0 else right_id
        raise ValueError(f"cosine vector has zero norm: {bad}")
    a = a / scale_a
    b = b / scale_b
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    value = float(np.dot(a / norm_a, b / norm_b))
    if not np.isfinite(value):
        raise ValueError(f"cosine result is non-finite: {left_id}, {right_id}")
    return value
