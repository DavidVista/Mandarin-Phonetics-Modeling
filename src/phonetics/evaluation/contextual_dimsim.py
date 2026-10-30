"""Full-range contextual DIMSIM scoring for immutable schema-3 bundles."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import distance_bin, spearman_result
from .bundle import load_evaluation_bundle
from .cosine import stable_cosine
from .representations import load_contextual_representations

MODES = ("dimsim-natural", "dimsim-balanced")
BIN_LABELS = tuple(f"[{x},{x + 10})" for x in range(0, 120, 10)) + ("[120,+inf)",)
MASKS = tuple(f"{i:03b}" for i in range(8))


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rho(rows: list[dict]) -> dict:
    return spearman_result([float(r["dimsim_distance"]) for r in rows],
                           [float(r["hidden_dissimilarity"]) for r in rows])


def score_contextual_dimsim(bundle: dict, occurrence_ids: list[str], hidden: np.ndarray,
                            provenance: dict, bundle_manifest_sha256: str,
                            analysis_config: dict | None = None) -> dict:
    """Score verified DIMSIM draws against raw contextual h.

    Returns both pair tables, full-range and overlap-stratum Spearman statistics,
    distance-bin diagnostics, and per-mode exploratory right-tail boundary curves.
    Invalid DIMSIM metrics remain in ``rows`` but are excluded from statistics.
    """
    cfg = {} if analysis_config is None else analysis_config
    if not isinstance(cfg, dict) or set(cfg) - {"min_unique_pairs_per_bin", "boundary_min_side_count", "boundary_thresholds"}:
        raise ValueError("analysis_config contains unsupported settings")
    bin_support = _positive_int(cfg.get("min_unique_pairs_per_bin", 100), "min_unique_pairs_per_bin")
    boundary_support = _positive_int(cfg.get("boundary_min_side_count", 500), "boundary_min_side_count")
    thresholds = cfg.get("boundary_thresholds", list(range(10, 111)))
    if not isinstance(thresholds, list) or not thresholds or any(type(x) is not int or not 10 <= x <= 110 for x in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("boundary_thresholds must be distinct integers from 10 through 110")
    if thresholds != sorted(thresholds):
        raise ValueError("boundary_thresholds must be sorted ascending")
    if not isinstance(bundle, dict) or type(bundle.get("manifest", {}).get("schema_version")) is not int or bundle["manifest"]["schema_version"] != 3:
        raise ValueError("a verified schema-3 bundle is required")
    if not isinstance(bundle_manifest_sha256, str) or len(bundle_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in bundle_manifest_sha256):
        raise ValueError("bundle_manifest_sha256 must be lowercase SHA-256")
    if not isinstance(provenance, dict) or provenance.get("bundle_manifest_sha256") != bundle_manifest_sha256:
        raise ValueError("representation provenance does not match bundle manifest")
    expected = sorted(r["occurrence_id"] for r in bundle["occurrences"])
    ids_are_valid = isinstance(occurrence_ids, list) and all(isinstance(x, str) and x for x in occurrence_ids)
    if not ids_are_valid or len(occurrence_ids) != len(set(occurrence_ids)) or sorted(occurrence_ids) != expected:
        actual = set(occurrence_ids) if ids_are_valid else set()
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"representation occurrence inventory mismatch; missing IDs={missing}, extra IDs={extra}")
    matrix = np.asarray(hidden)
    if matrix.ndim != 2 or matrix.shape[0] != len(occurrence_ids) or matrix.shape[1] == 0 or matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        bad_ids = [occurrence_ids[i] for i in range(min(len(occurrence_ids), matrix.shape[0] if matrix.ndim else 0)) if matrix.ndim == 2 and not np.isfinite(matrix[i]).all()]
        raise ValueError(f"hidden matrix must be a finite floating matrix; invalid IDs={bad_ids}")
    scales = np.max(np.abs(matrix.astype(np.float64)), axis=1)
    if np.any(scales == 0):
        raise ValueError(f"hidden matrix has zero-norm IDs={[occurrence_ids[i] for i in np.flatnonzero(scales == 0)]}")
    position = {key: i for i, key in enumerate(occurrence_ids)}
    output = {}
    for mode in MODES:
        raw = bundle.get("pairs", {}).get(mode)
        if not isinstance(raw, list):
            raise ValueError(f"bundle lacks {mode} pair rows")
        rows, draw_ids, pair_ids = [], set(), set()
        for pair in raw:
            draw_id = pair.get("draw_id")
            if not isinstance(draw_id, str) or not draw_id or draw_id in draw_ids:
                raise ValueError(f"missing or duplicate draw_id in {mode}: {draw_id}")
            draw_ids.add(draw_id)
            left_id, right_id = pair.get("left_id"), pair.get("right_id")
            if left_id not in position or right_id not in position or left_id == right_id:
                raise ValueError(f"invalid DIMSIM occurrence IDs: {left_id}, {right_id}")
            left, right = matrix[position[left_id]], matrix[position[right_id]]
            cosine = stable_cosine(left, right, left_id=left_id, right_id=right_id)
            if not math.isfinite(cosine):
                raise ValueError(f"non-finite cosine for IDs: {left_id}, {right_id}")
            distance = pair.get("dimsim_distance")
            status, label = distance_bin(distance)
            metric_status = pair.get("distance_status")
            valid = metric_status == "ok" and status == "ok"
            reason = "ok" if valid else (metric_status if metric_status != "ok" else status)
            row = {**pair, "cosine_similarity": cosine, "hidden_dissimilarity": 1.0 - cosine,
                   "analysis_status": reason, "distance_bin": label}
            rows.append(row)
            pair_ids.add(pair["pair_id"])
        valid = [r for r in rows if r["analysis_status"] == "ok"]
        strata = {mask: _rho([r for r in valid if r.get("overlap_mask") == mask]) for mask in MASKS}
        strata["no_zero_initial"] = _rho([r for r in valid if r.get("no_zero_initial_diagnostic") is True])
        bins = []
        for label in BIN_LABELS:
            selected = [r for r in valid if r["distance_bin"] == label]
            # pair_id is the unordered occurrence-pair identity in both modes;
            # balanced draw multiplicity remains in draw_count, not unique count.
            unique = len({r["pair_id"] for r in selected})
            bins.append({"bin": label, "draw_count": len(selected), "unique_pair_count": unique,
                         "mean_cosine": float(np.mean([r["cosine_similarity"] for r in selected])) if selected else None,
                         "supported": unique >= bin_support})
        boundary = []
        n = len(valid)
        min_side = max(boundary_support, math.ceil(.05 * n))
        for threshold in thresholds:
            left = [r for r in valid if float(r["dimsim_distance"]) < threshold]
            right = [r for r in valid if float(r["dimsim_distance"]) >= threshold]
            enough = len(left) >= min_side and len(right) >= min_side
            tail = _rho(right) if enough else {"rho": None, "valid_pair_count": len(right), "undefined_reason": "insufficient_boundary_support"}
            boundary.append({"threshold": threshold, "left_count": len(left), "right_count": len(right),
                             "left_rho": _rho(left)["rho"] if enough else None, "right_rho": tail["rho"],
                             "status": "ok" if enough and tail["rho"] is not None else (tail["undefined_reason"] or "insufficient_boundary_support")})
        candidates = [x for x in boundary if x["right_rho"] is not None]
        selected = min(candidates, key=lambda x: (abs(x["right_rho"]), x["threshold"])) if candidates else None
        output[mode] = {"rows": rows, "summary": {"draw_count": len(rows), "valid_draw_count": len(valid),
            "invalid_metric_count": len(rows)-len(valid), "unique_draw_count": len(draw_ids),
            "unique_pair_count": len(pair_ids), "valid_unique_pair_count": len({r["pair_id"] for r in valid}), "rho": _rho(valid),
            "strata": strata, "bins": bins, "boundary": {"min_side_count": min_side, "curve": boundary,
            "threshold": selected["threshold"] if selected else None,
            "status": "ok" if selected else ("no_valid_boundary_candidate" if n else "no_valid_metrics")}}}
    return {"schema_version": 1, "model_id": provenance.get("model_id"), "seed": provenance.get("seed"),
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "contract": {"modes": list(MODES), "full_range_rho": True, "bin_width": 10,
                         "min_unique_pairs_per_bin": bin_support, "boundary_thresholds": thresholds,
                         "boundary_min_side_count": boundary_support, "boundary_min_side_fraction": 0.05,
                         "boundary_rule": "exploratory data-selected threshold minimizing absolute right-tail Spearman; ties choose smaller threshold",
                         "precision": "full", "bootstrap": False}, "modes": output}


def score_contextual_dimsim_paths(bundle_path: str | Path, representation_path: str | Path,
                                  analysis_config: dict | None = None) -> dict:
    """Load verified bundle and representations, then call the pure scorer."""
    bundle = load_evaluation_bundle(bundle_path)
    bundle_sha = hashlib.sha256((Path(bundle_path) / "manifest.json").read_bytes()).hexdigest()
    loaded = load_contextual_representations(representation_path)
    return score_contextual_dimsim(bundle, loaded["occurrence_ids"], loaded["hidden"],
                                   loaded["manifest"]["provenance"], bundle_sha, analysis_config)
