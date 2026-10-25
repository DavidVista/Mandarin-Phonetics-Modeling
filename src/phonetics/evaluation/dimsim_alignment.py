"""Compare DIMSIM decompositions, distances, and inventory sensitivity.

The core consumes validated bundle dictionaries and accepts injected scorers.
File access is restricted to the explicit runner and export helpers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import csv
import gzip
import shutil
import tempfile
import re
import os
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from dimsim.utils.pinyin import Pinyin as AuthorPinyin

from phonetics.evaluation.dimsim import DimsimDistance
from phonetics.evaluation.dimsim_pypinyin import PypinyinDimsimDistance, score_original_components
from phonetics.pinyin import parse_components

PINNED_EVALUATION_MANIFEST_SHA256 = "1fbd485b38931369221e38dd30be5ea186de883b6c1fe7ace7000855645f43a2"
_RUN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


CONTRACTIONS = {frozenset(pair) for pair in (("iu", "iou"), ("ui", "uei"), ("un", "uen"))}
CATEGORIES = ("exact", "equivalent_umlaut", "u_vs_umlaut", "contracted_final", "other", "parse_error")


def _normal(value: str | None) -> str | None:
    return value.replace("ü", "v").replace("u:", "v") if isinstance(value, str) else value


def _raw_author(parser: Callable[[str], object], label: str) -> dict[str, object]:
    value = parser(label)
    initial, final, tone = value.consonant, value.vowel, value.tone
    if not isinstance(initial, str) or not isinstance(final, str) or not final or tone not in range(1, 6):
        raise ValueError("invalid author parser components")
    return {"initial": initial, "final": final, "tone": str(tone), "object": value}


def _raw_strict(parser: Callable[[str], object], label: str) -> dict[str, object]:
    value = parser(label)
    initial, final, tone = value.initial, value.final, value.tone
    if not isinstance(initial, str) or not isinstance(final, str) or not final or str(tone) not in tuple("12345"):
        raise ValueError("invalid strict parser components")
    return {"initial": initial, "final": final, "tone": str(tone), "object": value}


def extract_inventory(bundle: dict) -> list[str]:
    """Extract every unique syllable label represented by groups or occurrences."""

    group_labels = {row["pinyin"] for row in bundle.get("groups", ()) if row.get("group_kind") == "syllable"}
    occurrence_labels = {row["pinyin"] for row in bundle.get("occurrences", ())}
    labels = group_labels | occurrence_labels
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("bundle has no valid syllable inventory")
    return sorted(labels)


def build_decompositions(
    bundle: dict,
    *,
    author_parser: Callable[[str], object] = AuthorPinyin,
    strict_parser: Callable[[str], object] = parse_components,
) -> list[dict]:
    """Build lossless A/B decomposition rows and exclusive categories."""

    rows = []
    for label in extract_inventory(bundle):
        error_a = error_b = None
        try:
            a = _raw_author(author_parser, label)
        except Exception as exc:  # external parser diagnostics are data
            a = {"initial": None, "final": None, "tone": None, "object": None}
            error_a = f"{type(exc).__name__}:{exc}"
        try:
            b = _raw_strict(strict_parser, label)
        except Exception as exc:
            b = {"initial": None, "final": None, "tone": None, "object": None}
            error_b = f"{type(exc).__name__}:{exc}"
        norm_a = (_normal(a["initial"]), _normal(a["final"]), a["tone"])
        norm_b = (_normal(b["initial"]), _normal(b["final"]), b["tone"])
        flags = {
            "initial_changed": a["initial"] != b["initial"],
            "final_changed": a["final"] != b["final"],
            "tone_changed": a["tone"] != b["tone"],
            "normalized_initial_changed": norm_a[0] != norm_b[0],
            "normalized_final_changed": norm_a[1] != norm_b[1],
        }
        if error_a or error_b:
            base_category = "parse_error"
        elif (a["initial"], a["final"], a["tone"]) == (b["initial"], b["final"], b["tone"]):
            base_category = "exact"
        elif norm_a == norm_b:
            base_category = "equivalent_umlaut"
        elif frozenset((norm_a[1], norm_b[1])) == frozenset(("u", "v")):
            base_category = "u_vs_umlaut"
        elif frozenset((norm_a[1], norm_b[1])) in CONTRACTIONS:
            base_category = "contracted_final"
        else:
            base_category = "other"
        prefixes = [name for name in ("initial", "tone") if flags[f"{name}_changed"]]
        category = "+".join((*prefixes, base_category)) if prefixes and base_category not in {"exact", "parse_error"} else base_category
        rows.append({
            "label": label, "a_initial": a["initial"], "a_final": a["final"], "a_tone": a["tone"],
            "b_initial": b["initial"], "b_final": b["final"], "b_tone": b["tone"],
            "normalized_a_initial": norm_a[0], "normalized_a_final": norm_a[1],
            "normalized_b_initial": norm_b[0], "normalized_b_final": norm_b[1],
            "a_status": "ok" if error_a is None else error_a,
            "b_status": "ok" if error_b is None else error_b,
            "a_initial_present": bool(a["initial"]) if error_a is None else None,
            "b_initial_present": bool(b["initial"]) if error_b is None else None,
            "category": category, "difference_flags": flags,
            "difference_signature": "+".join(name for name, changed in flags.items() if changed) or "none",
        })
    return rows


def _pair_id(left: str, right: str) -> str:
    endpoints = sorted((left, right))
    return hashlib.sha256(json.dumps(endpoints, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def eligible_labels(rows: list[dict]) -> list[str]:
    """Return the numerical population: successfully parsed, B-nonzero labels."""

    return sorted(row["label"] for row in rows if row["b_status"] == "ok" and row["b_initial_present"] is True)


def affected_pair_plan(rows: list[dict]) -> list[tuple[str, str]]:
    """Return the stable unordered union involving every raw-discrepant label."""

    eligible = eligible_labels(rows)
    affected = {row["label"] for row in rows if row["label"] in eligible and row["category"] != "exact"}
    pairs = [(left, right) for left, right in itertools.combinations(eligible, 2) if left in affected or right in affected]
    expected = len(affected) * (len(eligible) - 1) - len(affected) * (len(affected) - 1) // 2
    if len(pairs) != expected:
        raise AssertionError("affected-pair union cardinality is inconsistent")
    return pairs


def _call(scorer: Callable[[str, str], float], left: str, right: str) -> tuple[float | None, str]:
    try:
        value = float(scorer(left, right))
        if not math.isfinite(value):
            return None, "nonfinite"
        return value, "ok"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def evaluate_alignment(
    decompositions: list[dict],
    *,
    scorer_a: Callable[[str, str], float] | None = None,
    scorer_b: Callable[[str, str], float] | None = None,
    reproduction_scorer: Callable[[object, object], float] = score_original_components,
    author_parser: Callable[[str], object] = AuthorPinyin,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> dict[str, object]:
    """Evaluate affected pairs and full-population directional symmetry."""

    scorer_a = scorer_a or DimsimDistance()
    scorer_b = scorer_b or PypinyinDimsimDistance()
    by_label = {row["label"]: row for row in decompositions}
    population = eligible_labels(decompositions)
    all_pairs = list(itertools.combinations(population, 2))

    def evaluate(left: str, right: str) -> dict:
        values, statuses = {}, {}
        for name, scorer, a, b in (
            ("a_forward", scorer_a, left, right), ("a_reverse", scorer_a, right, left),
            ("b_forward", scorer_b, left, right), ("b_reverse", scorer_b, right, left),
        ):
            values[name], statuses[name] = _call(scorer, a, b)
        reproduction_ok = True
        reproduction = {}
        for name, a, b in (("forward", left, right), ("reverse", right, left)):
            try:
                reproduced = float(reproduction_scorer(author_parser(a), author_parser(b)))
                reference = values[f"a_{name}"]
                if reference is None or not math.isfinite(reproduced) or not math.isclose(reproduced, reference, abs_tol=atol, rel_tol=rtol):
                    reproduction_ok = False
                    reproduction[f"a_reproduction_{name}_status"] = "mismatch"
                else:
                    reproduction[f"a_reproduction_{name}_status"] = "ok"
                reproduction[f"a_reproduced_{name}"] = reproduced if math.isfinite(reproduced) else None
            except Exception as exc:
                reproduction_ok = False
                reproduction[f"a_reproduction_{name}_status"] = f"error:{type(exc).__name__}"
                reproduction[f"a_reproduced_{name}"] = None
        joint = all(status == "ok" for status in statuses.values()) and reproduction_ok
        row = {
            "pair_id": _pair_id(left, right), "left": left, "right": right,
            "left_category": by_label[left]["category"], "right_category": by_label[right]["category"],
            **values, **{f"{key}_status": value for key, value in statuses.items()},
            **reproduction, "joint_valid": joint,
        }
        if joint:
            row.update({
                "delta_forward": values["b_forward"] - values["a_forward"],
                "delta_reverse": values["b_reverse"] - values["a_reverse"],
                "e_a": abs(values["a_forward"] - values["a_reverse"]),
                "e_b": abs(values["b_forward"] - values["b_reverse"]),
            })
        else:
            row.update({"delta_forward": None, "delta_reverse": None, "e_a": None, "e_b": None})
        return row

    evaluated = {pair: evaluate(*pair) for pair in all_pairs}
    tolerance = lambda x, y: math.isclose(x, y, abs_tol=atol, rel_tol=rtol)
    exceptions = [row for row in evaluated.values() if not row["joint_valid"] or not (
        tolerance(row["a_forward"], row["a_reverse"]) and tolerance(row["b_forward"], row["b_reverse"])
    )]
    valid = [row for row in evaluated.values() if row["joint_valid"]]
    asym = [row for row in valid if not (tolerance(row["a_forward"], row["a_reverse"]) and tolerance(row["b_forward"], row["b_reverse"]))]
    summary = {
        "planned": len(all_pairs), "valid": len(valid), "invalid": len(all_pairs) - len(valid),
        "a_asymmetric": sum(not tolerance(row["a_forward"], row["a_reverse"]) for row in valid),
        "b_asymmetric": sum(not tolerance(row["b_forward"], row["b_reverse"]) for row in valid),
        "a_asymmetric_fraction_of_valid": sum(not tolerance(row["a_forward"], row["a_reverse"]) for row in valid) / len(valid) if valid else None,
        "b_asymmetric_fraction_of_valid": sum(not tolerance(row["b_forward"], row["b_reverse"]) for row in valid) / len(valid) if valid else None,
        "mean_e_a": sum(row["e_a"] for row in valid) / len(valid) if valid else None,
        "max_e_a": max((row["e_a"] for row in valid), default=None),
        "mean_e_b": sum(row["e_b"] for row in valid) / len(valid) if valid else None,
        "max_e_b": max((row["e_b"] for row in valid), default=None),
        "invalid_status_counts": dict(sorted(Counter(
            status for row in evaluated.values() if not row["joint_valid"]
            for key, status in row.items() if key.endswith("_status") and status != "ok"
        ).items())),
    }
    return {
        "affected_pairs": [evaluated[pair] for pair in affected_pair_plan(decompositions)],
        "symmetry_summary": summary,
        "symmetry_exceptions": exceptions,
    }


def weighted_spearman(x, y, weights) -> tuple[float | None, str]:
    """Weighted Spearman with midranks equivalent to integer replication."""

    x, y, weights = map(lambda values: np.asarray(values, dtype=float), (x, y, weights))
    if not (x.ndim == y.ndim == weights.ndim == 1 and len(x) == len(y) == len(weights)):
        raise ValueError("weighted Spearman inputs must be equal-length vectors")
    if np.any(weights < 0) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(weights)):
        raise ValueError("weighted Spearman inputs must be finite with nonnegative weights")
    positive = weights > 0
    x, y, weights = x[positive], y[positive], weights[positive]
    if not len(x):
        return None, "zero_weight"

    def midranks(values):
        _, inverse = np.unique(values, return_inverse=True)
        group_weights = np.bincount(inverse, weights=weights)
        starts = np.cumsum(group_weights) - group_weights
        return (starts + (group_weights + 1.0) / 2.0)[inverse]

    rx, ry = midranks(x), midranks(y)
    total = float(weights.sum())
    mx, my = float(np.dot(weights, rx) / total), float(np.dot(weights, ry) / total)
    vx, vy = float(np.dot(weights, (rx - mx) ** 2)), float(np.dot(weights, (ry - my) ** 2))
    if vx == 0.0 or vy == 0.0:
        return None, "constant"
    return float(np.dot(weights, (rx - mx) * (ry - my)) / math.sqrt(vx * vy)), "ok"


def _prepare_scope(observations: list[dict]) -> dict:
    valid = [row for row in observations if row["valid"]]
    a = np.asarray([row["a"] for row in valid], dtype=float)
    b = np.asarray([row["b"] for row in valid], dtype=float)
    return {
        "a": a, "b": b, "e_a": np.asarray([row["pair"]["e_a"] for row in valid], dtype=float),
        "e_b": np.asarray([row["pair"]["e_b"] for row in valid], dtype=float),
        "a_forward": np.asarray([row["pair"]["a_forward"] for row in valid], dtype=float),
        "a_reverse": np.asarray([row["pair"]["a_reverse"] for row in valid], dtype=float),
        "b_forward": np.asarray([row["pair"]["b_forward"] for row in valid], dtype=float),
        "b_reverse": np.asarray([row["pair"]["b_reverse"] for row in valid], dtype=float),
        "pair_ids": [row["pair"]["pair_id"] for row in valid],
        "rank_groups_a": np.unique(a, return_inverse=True)[1],
        "rank_groups_b": np.unique(b, return_inverse=True)[1],
    }


def _cached_spearman(prepared: dict, weights: np.ndarray) -> tuple[float | None, str]:
    positive = weights > 0
    if not np.any(positive): return None, "zero_weight"
    weights = weights[positive]
    groups_a, groups_b = prepared["rank_groups_a"][positive], prepared["rank_groups_b"][positive]
    def ranks(groups):
        group_weights = np.bincount(groups, weights=weights)
        starts = np.cumsum(group_weights) - group_weights
        return (starts + (group_weights + 1.0) / 2.0)[groups]
    rx, ry = ranks(groups_a), ranks(groups_b); total = weights.sum()
    mx, my = np.dot(weights, rx) / total, np.dot(weights, ry) / total
    vx, vy = np.dot(weights, (rx-mx)**2), np.dot(weights, (ry-my)**2)
    if vx == 0 or vy == 0: return None, "constant"
    return float(np.dot(weights, (rx-mx)*(ry-my)) / math.sqrt(vx*vy)), "ok"


def _observations(pairs: list[dict], use_both_directions: bool) -> list[dict]:
    observations = []
    for pair in pairs:
        directions = ("forward", "reverse") if use_both_directions else ("forward",)
        for direction in directions:
            observations.append({
                "pair": pair, "direction": direction, "a": pair[f"a_{direction}"],
                "b": pair[f"b_{direction}"], "valid": pair["joint_valid"],
            })
    return observations


_CI_METRICS = ("mean_a", "mean_b", "mean_delta", "mean_abs_delta", "spearman", "mean_e_a", "mean_e_b",
               "changed_fraction", "asymmetric_fraction_a", "asymmetric_fraction_b")


def _close_array(left, right, *, atol: float, rtol: float):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return np.abs(left - right) <= np.maximum(atol, rtol * np.maximum(np.abs(left), np.abs(right)))


def _metrics(observations: list[dict], pair_weights: dict[str, float] | None = None,
             *, atol: float = 1e-12, rtol: float = 1e-10) -> tuple[dict, dict]:
    prepared = _prepare_scope(observations)
    weights = np.asarray([1.0 if pair_weights is None else pair_weights.get(pair_id, 1.0)
                          for pair_id in prepared["pair_ids"]])
    return _metrics_prepared(prepared, weights, atol=atol, rtol=rtol)


def _metrics_prepared(prepared: dict, weights: np.ndarray, *, atol: float, rtol: float) -> tuple[dict, dict]:
    full_weights = np.asarray(weights, dtype=float)
    positive = weights > 0
    if not np.any(positive):
        names = (*_CI_METRICS, "max_abs_delta", "max_e_a", "max_e_b")
        return {name: None for name in names}, {name: "zero_valid_weight" for name in names}
    a, b, weights = prepared["a"][positive], prepared["b"][positive], weights[positive]
    delta = b - a
    result = {
        "mean_a": float(np.average(a, weights=weights)), "mean_b": float(np.average(b, weights=weights)),
        "mean_delta": float(np.average(delta, weights=weights)),
        "mean_abs_delta": float(np.average(np.abs(delta), weights=weights)),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }
    e_a, e_b = prepared["e_a"][positive], prepared["e_b"][positive]
    a_forward, a_reverse = prepared["a_forward"][positive], prepared["a_reverse"][positive]
    b_forward, b_reverse = prepared["b_forward"][positive], prepared["b_reverse"][positive]
    result.update({
        "mean_e_a": float(np.average(e_a, weights=weights)), "max_e_a": float(np.max(e_a)),
        "mean_e_b": float(np.average(e_b, weights=weights)), "max_e_b": float(np.max(e_b)),
        "changed_fraction": float(np.average(~_close_array(a, b, atol=atol, rtol=rtol), weights=weights)),
        "asymmetric_fraction_a": float(np.average(~_close_array(a_forward, a_reverse, atol=atol, rtol=rtol), weights=weights)),
        "asymmetric_fraction_b": float(np.average(~_close_array(b_forward, b_reverse, atol=atol, rtol=rtol), weights=weights)),
    })
    result["spearman"], reason = _cached_spearman(prepared, full_weights)
    return result, {} if reason == "ok" else {"spearman": reason}


def summarize_statistics(
    decompositions: list[dict], alignment: dict[str, object], *,
    seed: int = 20261025, bootstrap_replicates: int = 1000, confidence: float = 0.95,
    atol: float = 1e-12, rtol: float = 1e-10,
) -> dict[str, object]:
    """Compute affected-union/category statistics and shared stratified CIs.

    Bootstrap strata cover the complete B-eligible inventory, including exact
    labels.  Invalid planned pairs remain in every fixed scope and merely carry
    zero contribution to comparative metrics.
    """

    if bootstrap_replicates < 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    pairs = alignment["affected_pairs"]
    summary = alignment["symmetry_summary"]
    use_both = bool(summary["a_asymmetric"] or summary["b_asymmetric"])
    observations = _observations(pairs, use_both)
    eligible = eligible_labels(decompositions)
    category_by_label = {row["label"]: row["category"] for row in decompositions}
    categories = sorted({category_by_label[label] for label in eligible})
    report_categories = [category for category in categories if category != "exact"]
    affected_labels = {row["label"] for row in decompositions if row["label"] in eligible and row["category"] != "exact"}
    scopes = {"all_affected": pairs}
    for category in report_categories:
        scopes[category] = [pair for pair in pairs if category in (pair["left_category"], pair["right_category"])]
    prepared_scopes = {scope: _prepare_scope(_observations(scope_pairs, use_both)) for scope, scope_pairs in scopes.items()}
    rows = []
    for scope, scope_pairs in scopes.items():
        metrics, reasons = _metrics(_observations(scope_pairs, use_both), atol=atol, rtol=rtol)
        rows.append({
            "scope": scope, "source_syllables": len(affected_labels) if scope == "all_affected" else sum(
                category_by_label[label] == scope for label in eligible),
            "planned_pairs": len(scope_pairs), "valid_pairs": sum(pair["joint_valid"] for pair in scope_pairs),
            "excluded_pairs": sum(not pair["joint_valid"] for pair in scope_pairs), **metrics,
            "point_undefined_reasons": reasons,
        })

    syllable_rows = []
    for focal in sorted(affected_labels):
        focal_pairs = [pair for pair in pairs if focal in (pair["left"], pair["right"])]
        directions = ("focal_to_partner", "partner_to_focal") if use_both else ("symmetric",)
        for direction in directions:
            focal_observations = []
            for pair in focal_pairs:
                focal_is_left = pair["left"] == focal
                selected = "forward" if direction == "symmetric" or (direction == "focal_to_partner") == focal_is_left else "reverse"
                focal_observations.append({"pair": pair, "direction": selected, "a": pair[f"a_{selected}"],
                                           "b": pair[f"b_{selected}"], "valid": pair["joint_valid"]})
            metrics, reasons = _metrics(focal_observations, atol=atol, rtol=rtol)
            syllable_rows.append({
                "syllable": focal, "category": category_by_label[focal], "direction": direction,
                "planned_pairs": len(focal_pairs), "valid_pairs": sum(pair["joint_valid"] for pair in focal_pairs),
                "excluded_pairs": sum(not pair["joint_valid"] for pair in focal_pairs), **metrics,
                "point_undefined_reasons": reasons,
            })

    strata = {category: [label for label in eligible if category_by_label[label] == category] for category in categories}
    rng = np.random.default_rng(seed)
    distributions = {scope: {name: [] for name in _CI_METRICS}
                     for scope in scopes}
    invalid_reasons = {scope: {name: Counter() for name in _CI_METRICS} for scope in scopes}
    for _ in range(bootstrap_replicates):
        multiplicity = {}
        for category, labels in strata.items():
            counts = rng.multinomial(len(labels), np.full(len(labels), 1.0 / len(labels)))
            multiplicity.update(zip(labels, counts.tolist()))
        weights = {pair["pair_id"]: multiplicity[pair["left"]] * multiplicity[pair["right"]] for pair in pairs}
        for scope, scope_pairs in scopes.items():
            prepared = prepared_scopes[scope]
            vector_weights = np.asarray([weights[pair_id] for pair_id in prepared["pair_ids"]], dtype=float)
            metrics, reasons = _metrics_prepared(prepared, vector_weights, atol=atol, rtol=rtol)
            for name in _CI_METRICS:
                value = metrics[name]
                if value is None:
                    invalid_reasons[scope][name][reasons.get(name, "undefined")] += 1
                else:
                    distributions[scope][name].append(value)
    alpha = (1.0 - confidence) / 2.0
    diagnostics = {}
    for row in rows:
        scope = row["scope"]
        diagnostics[scope] = {}
        for name, values in distributions[scope].items():
            row[f"{name}_ci_low"] = float(np.quantile(values, alpha)) if values else None
            row[f"{name}_ci_high"] = float(np.quantile(values, 1.0 - alpha)) if values else None
            diagnostics[scope][name] = {
                "valid_replicates": len(values), "invalid_replicates": bootstrap_replicates - len(values),
                "invalid_fraction": (bootstrap_replicates - len(values)) / bootstrap_replicates if bootstrap_replicates else None,
                "reasons": dict(sorted(invalid_reasons[scope][name].items())),
            }
    # Focal syllables stay fixed; their complete planned partner lists are
    # resampled.  A/B and directional observations share each partner draw.
    for focal in sorted(affected_labels):
        focal_rows = [row for row in syllable_rows if row["syllable"] == focal]
        focal_pairs = [pair for pair in pairs if focal in (pair["left"], pair["right"])]
        partners = [pair["right"] if pair["left"] == focal else pair["left"] for pair in focal_pairs]
        samples = {row["direction"]: {name: [] for name in _CI_METRICS} for row in focal_rows}
        reasons = {row["direction"]: {name: Counter() for name in _CI_METRICS} for row in focal_rows}
        focal_prepared = {}
        for row in focal_rows:
            observations_for_focal = []
            for pair in focal_pairs:
                focal_is_left = pair["left"] == focal
                selected = "forward" if row["direction"] == "symmetric" or (row["direction"] == "focal_to_partner") == focal_is_left else "reverse"
                observations_for_focal.append({"pair": pair, "direction": selected, "a": pair[f"a_{selected}"],
                                               "b": pair[f"b_{selected}"], "valid": pair["joint_valid"]})
            focal_prepared[row["direction"]] = _prepare_scope(observations_for_focal)
        for _ in range(bootstrap_replicates):
            counts = rng.multinomial(len(partners), np.full(len(partners), 1.0 / len(partners))) if partners else []
            weights = {pair["pair_id"]: count for pair, count in zip(focal_pairs, counts)}
            for row in focal_rows:
                prepared = focal_prepared[row["direction"]]
                vector_weights = np.asarray([weights[pair_id] for pair_id in prepared["pair_ids"]], dtype=float)
                metrics, why = _metrics_prepared(prepared, vector_weights, atol=atol, rtol=rtol)
                for name in _CI_METRICS:
                    value = metrics[name]
                    if value is None: reasons[row["direction"]][name][why.get(name, "undefined")] += 1
                    else: samples[row["direction"]][name].append(value)
        for row in focal_rows:
            row["bootstrap"] = {}
            for name, values in samples[row["direction"]].items():
                row[f"{name}_ci_low"] = float(np.quantile(values, alpha)) if values else None
                row[f"{name}_ci_high"] = float(np.quantile(values, 1 - alpha)) if values else None
                row["bootstrap"][name] = {"valid_replicates": len(values),
                                       "invalid_replicates": bootstrap_replicates - len(values),
                                       "invalid_fraction": (bootstrap_replicates - len(values)) / bootstrap_replicates if bootstrap_replicates else None,
                                       "reasons": dict(sorted(reasons[row["direction"]][name].items()))}
    total_pairs = len(eligible) * (len(eligible) - 1) // 2
    return {
        "statistics_by_category": rows,
        "statistics_by_syllable": syllable_rows,
        "affected_union_fraction": len(pairs) / total_pairs if total_pairs else None,
        "use_both_directions": use_both,
        "bootstrap_diagnostics": {"seed": seed, "replicates": bootstrap_replicates, "confidence": confidence,
                                  "metrics": diagnostics,
                                  "syllables": [{"syllable": row["syllable"], "direction": row["direction"],
                                                 "metrics": row["bootstrap"]} for row in syllable_rows]},
    }


def by_category(decompositions: list[dict], label: str) -> str:
    return next(row["category"] for row in decompositions if row["label"] == label)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_rows(path: Path, rows: list[dict], headers: list[str], *, compressed: bool = False) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else _json_safe(value) for key, value in row.items()})


_PAIR_HEADERS = ["pair_id", "left", "right", "left_category", "right_category",
                 "a_forward", "a_reverse", "b_forward", "b_reverse",
                 "a_forward_status", "a_reverse_status", "b_forward_status", "b_reverse_status",
                 "a_reproduction_forward_status", "a_reproduced_forward",
                 "a_reproduction_reverse_status", "a_reproduced_reverse", "joint_valid",
                 "delta_forward", "delta_reverse", "e_a", "e_b"]
_STAT_HEADERS = ["planned_pairs", "valid_pairs", "excluded_pairs", *_CI_METRICS,
                 "max_abs_delta", "max_e_a", "max_e_b", "point_undefined_reasons",
                 *(f"{metric}_ci_{bound}" for metric in _CI_METRICS for bound in ("low", "high"))]


def coverage_rows(decompositions: list[dict]) -> list[dict]:
    """Report category, parser, and initial-presence coverage with denominators."""
    populations = {"inventory": decompositions,
                   "b_nonzero": [row for row in decompositions
                                 if row["b_status"] == "ok" and row["b_initial_present"]]}
    output = []
    for population, rows in populations.items():
        denominator = len(rows)
        measures = {"total": denominator,
                    "initial_presence_disagreement": sum(
                        row["a_status"] == row["b_status"] == "ok"
                        and row["a_initial_present"] != row["b_initial_present"] for row in rows),
                    "raw_discrepant": sum(row["category"] != "exact" for row in rows),
                    "normalized_discrepant": sum(
                        row["category"] not in ("exact", "equivalent_umlaut") for row in rows)}
        measures.update({f"category:{key}": count for key, count in Counter(row["category"] for row in rows).items()})
        for side in ("a", "b"):
            measures.update({f"{side}_status:{key}": count
                             for key, count in Counter(row[f"{side}_status"] for row in rows).items()})
        output.extend({"population": population, "measure": key, "count": value,
                       "denominator": denominator, "fraction": value / denominator if denominator else None}
                      for key, value in sorted(measures.items()))
    return output


def export_alignment(output_dir: str | Path, *, config: dict, decompositions: list[dict],
                     alignment: dict, statistics: dict, provenance: dict) -> Path:
    """Atomically write the fixed alignment artifact set without overwriting."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        coverage = coverage_rows(decompositions)
        files = {
            "decompositions.csv": (decompositions, list(decompositions[0]) if decompositions else ["label", "category"]),
            "coverage.csv": (coverage, ["population", "measure", "count", "denominator", "fraction"]),
            "pairs.csv.gz": (alignment["affected_pairs"], _PAIR_HEADERS),
            "symmetry-summary.csv": ([alignment["symmetry_summary"]], list(alignment["symmetry_summary"])),
            "symmetry-exceptions.csv.gz": (alignment["symmetry_exceptions"], _PAIR_HEADERS),
            "statistics-by-syllable.csv": (statistics["statistics_by_syllable"], ["syllable", "category", "direction", *_STAT_HEADERS, "bootstrap"]),
            "statistics-by-category.csv": (statistics["statistics_by_category"], ["scope", "source_syllables", *_STAT_HEADERS]),
        }
        for name, (rows, headers) in files.items():
            _csv_rows(temporary / name, rows, headers, compressed=name.endswith(".gz"))
        (temporary / "config.json").write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "bootstrap-diagnostics.json").write_text(json.dumps(_json_safe(statistics["bootstrap_diagnostics"]), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        artifact_names = sorted([*files, "config.json", "bootstrap-diagnostics.json"])
        manifest = {"schema_version": 1, "provenance": provenance,
                    "analysis_summary": {"affected_union_fraction": statistics["affected_union_fraction"],
                                         "use_both_directions": statistics["use_both_directions"]}, "artifacts": {
            name: hashlib.sha256((temporary / name).read_bytes()).hexdigest() for name in artifact_names}}
        (temporary / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def run_alignment(config: dict) -> Path:
    """Validate the pinned input, compute the analysis, and atomically export it."""

    resolved = {"input_manifest_sha256": PINNED_EVALUATION_MANIFEST_SHA256, "seed": 20261025,
                "bootstrap_replicates": 1000, "confidence": 0.95, "atol": 1e-12, "rtol": 1e-10, **config}
    run_id = resolved.get("run_id")
    if (not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None or ".." in run_id
            or run_id == "replace-with-explicit-run-id"):
        raise ValueError("run_id must be a safe single-directory name")
    if (isinstance(resolved["bootstrap_replicates"], bool) or not isinstance(resolved["bootstrap_replicates"], int)
            or resolved["bootstrap_replicates"] <= 0 or isinstance(resolved["seed"], bool) or not isinstance(resolved["seed"], int)
            or resolved["seed"] < 0
            or not isinstance(resolved["confidence"], (int, float)) or not 0 < resolved["confidence"] < 1
            or any(isinstance(resolved[key], bool) or not isinstance(resolved[key], (int, float))
                   or not math.isfinite(resolved[key]) or resolved[key] < 0 for key in ("atol", "rtol"))):
        raise ValueError("invalid numerical analysis configuration")
    input_path, output_root = Path(resolved["input_path"]).resolve(), Path(resolved["output_root"]).resolve()
    target = output_root / run_id
    if target.exists() or target == input_path or target in input_path.parents or input_path in target.parents:
        raise ValueError("output must be new and separate from the immutable input")
    manifest_path = input_path / "manifest.json"
    expected = resolved["input_manifest_sha256"]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError("evaluation manifest SHA256 does not match the pinned input")
    from phonetics.evaluation.bundle import load_evaluation_bundle
    bundle = load_evaluation_bundle(input_path)
    decompositions = build_decompositions(bundle)
    alignment = evaluate_alignment(decompositions, atol=resolved["atol"], rtol=resolved["rtol"])
    statistics = summarize_statistics(decompositions, alignment, seed=resolved["seed"],
                                      bootstrap_replicates=resolved["bootstrap_replicates"], confidence=resolved["confidence"],
                                      atol=resolved["atol"], rtol=resolved["rtol"])
    source_paths = [Path(__file__), Path(__file__).with_name("dimsim.py"), Path(__file__).with_name("dimsim_pypinyin.py"),
                    Path(__file__).with_name("bundle.py"), Path(__file__).parents[1] / "pinyin.py"]
    provenance = {"input_manifest_sha256": expected, "source_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "scorer_a": DimsimDistance().provenance, "scorer_b": PypinyinDimsimDistance().provenance}
    resolved["analysis_summary"] = {"affected_union_fraction": statistics["affected_union_fraction"],
                                    "use_both_directions": statistics["use_both_directions"]}
    return export_alignment(target, config=resolved, decompositions=decompositions, alignment=alignment,
                            statistics=statistics, provenance=provenance)
