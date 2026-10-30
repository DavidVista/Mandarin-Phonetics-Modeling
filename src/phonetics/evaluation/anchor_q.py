"""Full precision scoring for fixed schema-3 contextual anchor-Q draws."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_sampling import EDGES
from .bundle import load_evaluation_bundle
from .cosine import stable_cosine
from .representations import load_contextual_representations

DEFAULT_TIE_TOLERANCE = 1e-12


def _validate_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("tie_tolerance must be a finite non-negative number")
    return float(value)


def score_anchor_q(bundle: dict[str, Any], occurrence_ids: list[str], hidden: np.ndarray,
                   provenance: dict[str, Any], *, bundle_manifest_sha256: str,
                   tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> dict[str, Any]:
    """Score all pre-sampled draws against raw contextual h.

    Inputs are a verified schema-3 bundle object, its representation occurrence ID
    inventory and raw matrix (rows may be in any order), validated representation
    provenance, and the SHA-256 digest of the already-verified bundle manifest.
    Returns JSON-compatible draw scores, edge/component/overall Q and
    MC standard errors, plus coverage and provenance. No sampling or h aggregation occurs.
    """
    tolerance = _validate_tolerance(tie_tolerance)
    manifest = bundle["manifest"]
    if manifest.get("schema_version") != 3:
        raise ValueError("anchor-Q scorer requires a schema-3 bundle")
    expected_hash = bundle_manifest_sha256
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)):
        raise ValueError("bundle manifest SHA-256 digest is invalid")
    if provenance.get("bundle_manifest_sha256") != expected_hash:
        raise ValueError("representation provenance bundle manifest hash mismatch")
    bundle_occurrences = [row["occurrence_id"] for row in bundle["occurrences"]]
    if len(occurrence_ids) != len(set(occurrence_ids)) or set(occurrence_ids) != set(bundle_occurrences):
        raise ValueError("representation occurrence ID inventory does not match bundle")
    matrix = np.asarray(hidden)
    if matrix.ndim != 2 or matrix.shape[0] != len(occurrence_ids):
        raise ValueError("hidden matrix rows do not match occurrence ID inventory")
    if matrix.dtype.kind != "f":
        raise ValueError("raw hidden matrix must have a floating-point dtype")
    positions = {oid: i for i, oid in enumerate(occurrence_ids)}
    sampling = bundle["config"]["sampling"]["anchor_q"]
    contract_tol = bundle["config"]["scoring"]["anchor_q"]["tie_tolerance"]
    if tolerance != contract_tol:
        raise ValueError("tie_tolerance differs from immutable bundle scoring contract")
    expected_edges = [f"{c}:{lo}->{hi}" for c, lo, hi in EDGES]
    if sampling["edge_order"] != expected_edges:
        raise ValueError("bundle anchor-Q edge order mismatch")
    per_edge = sampling["draws_per_edge"]
    rows = bundle["triplets"]
    if len(rows) != len(expected_edges) * per_edge:
        raise ValueError("anchor-Q draw count does not match bundle contract")
    counts = {edge: 0 for edge in expected_edges}
    draw_ids = set()
    scored = []
    for global_index, row in enumerate(rows):
        draw_id, edge = row["draw_id"], row["edge_id"]
        expected_edge = expected_edges[global_index // per_edge]
        if edge != expected_edge:
            raise ValueError(f"anchor-Q rows violate global edge block order at draw {draw_id}")
        if draw_id in draw_ids:
            raise ValueError(f"duplicate anchor-Q draw ID: {draw_id}")
        draw_ids.add(draw_id)
        if edge not in counts:
            raise ValueError(f"unexpected anchor-Q edge: {edge}")
        if len({row[k] for k in ("anchor_id", "high_id", "low_id")}) != 3:
            raise ValueError(f"triplet occurrence IDs must be distinct: {draw_id}")
        if (type(row.get("draw_index")) is not int or row["draw_index"] != counts[edge]):
            raise ValueError(f"anchor-Q draw index/order mismatch: {draw_id}")
        expected_draw_id = f"anchor-q:v1:{edge}:{counts[edge]:06d}"
        if draw_id != expected_draw_id:
            raise ValueError(f"anchor-Q draw ID/order mismatch: {draw_id}")
        counts[edge] += 1
        if counts[edge] > per_edge:
            raise ValueError(f"anchor-Q edge draw count exceeds contract: {edge}")
        a, h, l = (row[k] for k in ("anchor_id", "high_id", "low_id"))
        ca = stable_cosine(matrix[positions[a]], matrix[positions[h]], left_id=a, right_id=h)
        cb = stable_cosine(matrix[positions[a]], matrix[positions[l]], left_id=a, right_id=l)
        difference = ca - cb
        score = 0.5 if abs(difference) <= tolerance else (1.0 if difference > 0 else 0.0)
        scored.append({"draw_id": draw_id, "triplet_id": row["triplet_id"], "edge_id": edge,
                       "cosine_anchor_high": ca, "cosine_anchor_low": cb,
                       "difference": difference, "score": score})
    if any(counts[e] != per_edge for e in expected_edges):
        raise ValueError("anchor-Q edge count mismatch; complete coverage required")
    edge_results = {}
    edge_ses = {}
    for edge in expected_edges:
        values = np.asarray([r["score"] for r in scored if r["edge_id"] == edge], dtype=np.float64)
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        edge_ses[edge] = se
        edge_results[edge] = {"q": float(np.mean(values)), "mc_se": se,
                              "mc_se_status": "ok" if se is not None else "not_available_single_draw",
                              "draw_count": int(len(values))}
    components = {}
    for component in ("initial", "final", "tone"):
        edges = [f"{c}:{lo}->{hi}" for c, lo, hi in EDGES if c == component]
        components[component] = {"q": float(np.mean([edge_results[e]["q"] for e in edges])),
                                 "mc_se": (float(math.sqrt(sum(edge_ses[e] ** 2 for e in edges)) / len(edges))
                                           if all(edge_ses[e] is not None for e in edges) else None),
                                 "mc_se_status": "ok" if all(edge_ses[e] is not None for e in edges) else "not_available_single_draw"}
    overall_se = (float(math.sqrt(sum(edge_ses[e] ** 2 for e in expected_edges)) / len(expected_edges))
                  if all(edge_ses[e] is not None for e in expected_edges) else None)
    return {"schema_version": 1, "model_id": provenance["model_id"], "seed": provenance["seed"],
            "draw_scores": scored, "edges": edge_results, "components": components,
            "overall": {"q": float(np.mean([edge_results[e]["q"] for e in expected_edges])),
                        "mc_se": overall_se, "mc_se_status": "ok" if overall_se is not None else "not_available_single_draw"},
            "coverage": bundle["coverage"]["anchor_q"], "provenance": provenance,
            "bundle_manifest_sha256": expected_hash, "tie_tolerance": tolerance}


def score_anchor_q_paths(bundle_path: str | Path, representation_path: str | Path) -> dict[str, Any]:
    """Load validated schema-3 bundle and contextual representation paths, then score."""
    bundle = load_evaluation_bundle(bundle_path)
    if bundle["manifest"].get("schema_version") != 3:
        raise ValueError("anchor-Q path scorer requires a schema-3 bundle")
    bundle_manifest_sha256 = hashlib.sha256((Path(bundle_path) / "manifest.json").read_bytes()).hexdigest()
    representation = load_contextual_representations(representation_path)
    return score_anchor_q(bundle, representation["occurrence_ids"], representation["hidden"],
                          representation["manifest"]["provenance"],
                          bundle_manifest_sha256=bundle_manifest_sha256,
                          tie_tolerance=bundle["config"]["scoring"]["anchor_q"]["tie_tolerance"])
