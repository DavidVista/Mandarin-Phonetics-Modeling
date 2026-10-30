"""Validated five-seed summaries and plot-ready data for contextual DIMSIM."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..objectives import canonical_model_id
from .analysis import distance_bin, spearman_result
from .contextual_dimsim import BIN_LABELS, MASKS, MODES, score_contextual_dimsim
from .dimsim_sampling import pair_identity
from .bundle import load_evaluation_bundle
from .representations import load_contextual_representations
from .seed_runs import discover_seed_runs

SEEDS = (0, 1, 2, 3, 4)
_METRIC_FIELDS = ("cosine_similarity", "hidden_dissimilarity")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _stats(rows: list[dict]) -> dict:
    return spearman_result([float(r["dimsim_distance"]) for r in rows],
                           [float(r["hidden_dissimilarity"]) for r in rows])


def _assert_rho(actual, expected, location):
    if not isinstance(actual, dict) or actual.get("undefined_reason") != expected.get("undefined_reason") or actual.get("valid_pair_count") != expected.get("valid_pair_count"):
        raise ValueError(f"inconsistent rho metadata at {location}")
    value = actual.get("rho")
    if expected["rho"] is None:
        if value is not None: raise ValueError(f"unexpected defined rho at {location}")
    elif not _finite(value) or not math.isclose(float(value), expected["rho"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"rho does not match pair rows at {location}")


def _validate_score(score: dict, model: str, seed: int, bundle_hash: str):
    if (type(score.get("schema_version")) is not int or score["schema_version"] != 1
            or score.get("model_id") != model or type(score.get("seed")) is not int or score["seed"] != seed
            or score.get("bundle_manifest_sha256") != bundle_hash):
        raise ValueError(f"score identity mismatch for {model} seed {seed}")
    contract = score.get("contract")
    required_contract = {"modes", "full_range_rho", "bin_width", "min_unique_pairs_per_bin", "boundary_thresholds",
                         "boundary_min_side_count", "boundary_min_side_fraction", "boundary_rule", "precision", "bootstrap"}
    if (not isinstance(contract, dict) or set(contract) != required_contract or contract["modes"] != list(MODES)
            or contract["full_range_rho"] is not True or type(contract["bin_width"]) is not int or contract["bin_width"] != 10
            or type(contract["min_unique_pairs_per_bin"]) is not int or contract["min_unique_pairs_per_bin"] <= 0
            or type(contract["boundary_min_side_count"]) is not int or contract["boundary_min_side_count"] <= 0
            or contract["boundary_min_side_fraction"] != 0.05 or contract["precision"] != "full" or contract["bootstrap"] is not False):
        raise ValueError(f"invalid analysis contract for {model} seed {seed}")
    thresholds = contract["boundary_thresholds"]
    if (not isinstance(thresholds, list) or not thresholds or any(type(x) is not int or not 10 <= x <= 110 for x in thresholds)
            or thresholds != sorted(set(thresholds))):
        raise ValueError("boundary thresholds must be unique and sorted")
    modes = score.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(MODES):
        raise ValueError("score mode inventory is invalid")
    identities, mode_counts = {}, {}
    for mode in MODES:
        value = modes[mode]
        if not isinstance(value, dict) or set(value) != {"rows", "summary"} or not isinstance(value["rows"], list):
            raise ValueError(f"invalid mode rows for {model} seed {seed} {mode}")
        rows = value["rows"]
        if not rows: raise ValueError("DIMSIM mode cannot have an empty draw table")
        seen_draw, pair_ids, valid, bins = set(), set(), [], {label: [] for label in BIN_LABELS}
        shared_rows = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict): raise ValueError("DIMSIM row must be an object")
            draw_id, pair_id = row.get("draw_id"), row.get("pair_id")
            if (type(row.get("draw_index")) is not int or row["draw_index"] != index
                    or draw_id != f"{mode}:v1:{index:06d}" or draw_id in seen_draw
                    or not isinstance(pair_id, str) or not pair_id):
                raise ValueError(f"invalid draw/pair identity in {model} seed {seed} {mode}")
            seen_draw.add(draw_id); pair_ids.add(pair_id)
            left, right, mask = row.get("left_id"), row.get("right_id"), row.get("overlap_mask")
            if not all(isinstance(x, str) and x for x in (left, right)) or left >= right or mask not in MASKS:
                raise ValueError("invalid endpoints or overlap mask")
            expected_mode = "natural" if mode.endswith("natural") else "balanced"
            if row.get("mode") != expected_mode or pair_id != pair_identity(left, right):
                raise ValueError("pair identity/mode metadata is invalid")
            status, distance = row.get("distance_status"), row.get("dimsim_distance")
            bin_status, label = distance_bin(distance)
            is_valid = status == "ok" and bin_status == "ok"
            expected_status = "ok" if is_valid else (status if status != "ok" else bin_status)
            if row.get("analysis_status") != expected_status or row.get("distance_bin") != label:
                raise ValueError("DIMSIM metric status/bin does not match source distance")
            if status == "ok":
                if not _finite(distance) or float(distance) < 0: raise ValueError("ok DIMSIM distance must be finite and nonnegative")
            elif distance is not None or not isinstance(status, str) or not (status == "not_computed" or status.startswith("invalid:")):
                raise ValueError("invalid DIMSIM metric status/value")
            cosine, dissimilarity = (row.get(k) for k in _METRIC_FIELDS)
            if not _finite(cosine) or not _finite(dissimilarity) or not math.isclose(float(dissimilarity), 1.0-float(cosine), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("cosine/dissimilarity values are invalid or inconsistent")
            if row.get("no_zero_initial_diagnostic") is not True:
                raise ValueError("zero-initial diagnostic flag is invalid")
            common = (draw_id, pair_id, index, left, right, mask, status, distance)
            if expected_mode == "balanced":
                first, second = row.get("first_id"), row.get("second_id")
                if {first, second} != {left, right} or first == second:
                    raise ValueError("balanced draw orientation does not match endpoints")
                common += (first, second, row.get("first_syllable_id"), row.get("second_syllable_id"))
            shared_rows.append(common)
            if is_valid:
                valid.append(row); bins[label].append(row)
        summary = value["summary"]
        if not isinstance(summary, dict): raise ValueError("mode summary is required")
        expected_counts = {"draw_count": len(rows), "valid_draw_count": len(valid), "invalid_metric_count": len(rows)-len(valid),
                           "unique_draw_count": len(seen_draw), "unique_pair_count": len(pair_ids),
                           "valid_unique_pair_count": len({r["pair_id"] for r in valid})}
        if any(type(summary.get(k)) is not int or summary[k] != v for k, v in expected_counts.items()):
            raise ValueError(f"mode count diagnostics mismatch for {model} seed {seed} {mode}")
        _assert_rho(summary.get("rho"), _stats(valid), f"{model}/{seed}/{mode}/full")
        strata = summary.get("strata")
        if not isinstance(strata, dict) or set(strata) != {*MASKS, "no_zero_initial"}:
            raise ValueError("overlap strata inventory is invalid")
        for key in (*MASKS, "no_zero_initial"):
            selected = [r for r in valid if (r["overlap_mask"] == key if key in MASKS else r.get("no_zero_initial_diagnostic") is True)]
            _assert_rho(strata[key], _stats(selected), f"{model}/{seed}/{mode}/{key}")
        bin_rows = summary.get("bins")
        if (not isinstance(bin_rows, list) or len(bin_rows) != len(BIN_LABELS)
                or any(not isinstance(x, dict) for x in bin_rows) or [x.get("bin") for x in bin_rows] != list(BIN_LABELS)):
            raise ValueError("distance bins are not canonical")
        for label, actual in zip(BIN_LABELS, bin_rows):
            selected = bins[label]; unique = len({r["pair_id"] for r in selected})
            mean = float(np.mean([r["cosine_similarity"] for r in selected])) if selected else None
            if (type(actual.get("draw_count")) is not int or actual["draw_count"] != len(selected)
                    or type(actual.get("unique_pair_count")) is not int or actual["unique_pair_count"] != unique
                    or actual.get("supported") is not (unique >= contract["min_unique_pairs_per_bin"])):
                raise ValueError("bin count/support diagnostics mismatch")
            av = actual.get("mean_cosine")
            if mean is None:
                if av is not None: raise ValueError("empty bin mean must be null")
            elif not _finite(av) or not math.isclose(float(av), mean, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("bin cosine mean does not match pair rows")
        boundary = summary.get("boundary")
        if not isinstance(boundary, dict) or type(boundary.get("min_side_count")) is not int:
            raise ValueError("boundary summary is invalid")
        minimum = max(contract["boundary_min_side_count"], math.ceil(.05 * len(valid)))
        if boundary["min_side_count"] != minimum: raise ValueError("boundary side support differs from contract")
        curve = boundary.get("curve")
        if (not isinstance(curve, list) or any(not isinstance(r, dict) for r in curve)
                or [r.get("threshold") for r in curve] != thresholds):
            raise ValueError("boundary curve threshold grid differs from contract")
        computed = []
        for item in curve:
            t = item["threshold"]
            lo = [r for r in valid if r["dimsim_distance"] < t]
            hi = [r for r in valid if r["dimsim_distance"] >= t]
            enough = len(lo) >= minimum and len(hi) >= minimum
            lr, rr = (_stats(lo)["rho"] if enough else None), (_stats(hi)["rho"] if enough else None)
            status = "ok" if enough and rr is not None else ("insufficient_boundary_support" if not enough else _stats(hi)["undefined_reason"])
            if (item.get("left_count") != len(lo) or item.get("right_count") != len(hi)
                    or item.get("left_rho") != lr or item.get("right_rho") != rr or item.get("status") != status):
                raise ValueError("boundary curve does not match original valid rows")
            if rr is not None: computed.append((abs(rr), t))
        chosen = min(computed)[1] if computed else None
        expected_boundary_status = "ok" if chosen is not None else ("no_valid_boundary_candidate" if valid else "no_valid_metrics")
        if boundary.get("threshold") != chosen or boundary.get("status") != expected_boundary_status:
            raise ValueError("selected boundary does not match boundary curve")
        identities[mode] = shared_rows
        mode_counts[mode] = {**expected_counts, "valid_unique_pair_count": expected_counts["valid_unique_pair_count"]}
    return contract, identities, mode_counts


def _summary(values: list[float | None], value_name: str = "value", seeds=SEEDS) -> dict:
    labelled = [{"seed": seed, value_name: value} for seed, value in zip(seeds, values)]
    if len(values) != 5 or any(value is None for value in values):
        return {"seed_values": labelled, "mean": None, "sample_sd": None, "status": "incomplete_five_valid_values"}
    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vals).all(): raise ValueError("summary values must be finite")
    return {"seed_values": labelled, "mean": float(np.mean(vals)), "sample_sd": float(np.std(vals, ddof=1)), "status": "ok"}


def build_dimsim_report(batch_result: dict, reference_model: str = "syllable") -> dict:
    """Validate cross-seed pair and analysis identity, then build complete summaries."""
    if not isinstance(batch_result, dict) or type(batch_result.get("schema_version")) is not int or batch_result["schema_version"] != 1:
        raise ValueError("batch schema_version must be integer 1")
    bundle_hash = batch_result.get("bundle_manifest_sha256")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64 or any(c not in "0123456789abcdef" for c in bundle_hash):
        raise ValueError("invalid bundle manifest hash")
    reference = canonical_model_id(reference_model)
    records = batch_result.get("results")
    if not isinstance(records, list) or not records: raise ValueError("batch results are required")
    grouped, shared_contract, identity_reference, counts_reference = {}, None, None, None
    for rec in records:
        if not isinstance(rec, dict): raise ValueError("batch record must be an object")
        model, seed = rec.get("model_id"), rec.get("seed")
        if not isinstance(model, str) or canonical_model_id(model) != model or type(seed) is not int or seed not in SEEDS:
            raise ValueError("invalid model/seed identity")
        if (model, seed) in grouped: raise ValueError("duplicate model/seed result")
        score = rec.get("score")
        if not isinstance(score, dict): raise ValueError("batch score must be an object")
        contract, identities, mode_counts = _validate_score(score, model, seed, bundle_hash)
        if shared_contract is None: shared_contract, identity_reference, counts_reference = contract, identities, mode_counts
        elif contract != shared_contract or identities != identity_reference or mode_counts != counts_reference:
            raise ValueError("scores differ in analysis contract, pair/draw metadata, counts, or bin support")
        grouped[(model, seed)] = rec["score"]
    models = sorted({m for m, _ in grouped})
    if reference not in models: raise ValueError("reference model is missing")
    tables = {"seed_rho": [], "model_rho": [], "seed_boundary": [], "model_boundary": [], "paired_rho_deltas": [], "bins": [], "mode_counts": []}
    plot = {"curves": [], "boundaries": [], "mode_counts": []}
    for mode in MODES:
        tables["mode_counts"].append({"mode": mode, **counts_reference[mode]})
        plot["mode_counts"].append({"mode": mode, **counts_reference[mode]})
    for model in models:
        for mode in MODES:
            all5 = [grouped.get((model, seed), {}).get("modes", {}).get(mode) for seed in SEEDS]
            rho_vals = [x["summary"]["rho"]["rho"] if x else None for x in all5]
            for seed, item, rho in zip(SEEDS, all5, rho_vals):
                s = item["summary"] if item else None
                tables["seed_rho"].append({"model_id": model, "seed": seed, "mode": mode, "rho": rho,
                    "draw_count": s["draw_count"] if s else None, "valid_draw_count": s["valid_draw_count"] if s else None,
                    "invalid_metric_count": s["invalid_metric_count"] if s else None,
                    "unique_pair_count": s["unique_pair_count"] if s else None,
                    "undefined_reason": s["rho"]["undefined_reason"] if s else "missing_seed",
                    "status": "ok" if rho is not None else ("missing_seed" if s is None else "undefined_rho")})
            tables["model_rho"].append({"model_id": model, "mode": mode, **_summary(rho_vals, "rho")})
            curves = []
            for i, label in enumerate(BIN_LABELS):
                bin_seed = [item["summary"]["bins"][i] if item else None for item in all5]
                means = [x["mean_cosine"] if x else None for x in bin_seed]
                support = all(x is not None and x["supported"] for x in bin_seed)
                row = {"model_id": model, "mode": mode, "bin": label,
                       "mean_cosine": float(np.mean(means)) if all(x is not None for x in means) else None,
                       "seed_points": [{"seed": seed, "mean_cosine": means[j],
                                        "draw_count": bin_seed[j]["draw_count"] if bin_seed[j] else None,
                                        "unique_pair_count": bin_seed[j]["unique_pair_count"] if bin_seed[j] else None}
                                       for j, seed in enumerate(SEEDS)], "supported": support, "connect_to_next": False}
                curves.append(row)
            for i in range(len(curves)-1): curves[i]["connect_to_next"] = curves[i]["supported"] and curves[i+1]["supported"]
            tables["bins"].extend(curves); plot["curves"].extend(curves)
            thresholds = [item["summary"]["boundary"]["threshold"] if item else None for item in all5]
            for seed, item, threshold in zip(SEEDS, all5, thresholds):
                tables["seed_boundary"].append({"model_id": model, "seed": seed, "mode": mode, "threshold": threshold,
                    "status": item["summary"]["boundary"]["status"] if item else "missing_seed"})
            bsummary = _summary(thresholds, "threshold")
            median = float(np.median(thresholds)) if bsummary["status"] == "ok" else None
            tables["model_boundary"].append({"model_id": model, "mode": mode, **bsummary, "median": median,
                                             "median_status": "ok" if median is not None else "requires_all_five_valid"})
            plot["boundaries"].append({"model_id": model, "mode": mode, "median": median,
                                       "seed_points": [{"seed": s, "threshold": t} for s, t in zip(SEEDS, thresholds)]})
    for model in models:
        if model == reference: continue
        for mode in MODES:
            deltas = []
            for seed in SEEDS:
                a_score, b_score = grouped.get((model, seed)), grouped.get((reference, seed))
                a = a_score["modes"][mode]["summary"]["rho"]["rho"] if a_score else None
                b = b_score["modes"][mode]["summary"]["rho"]["rho"] if b_score else None
                deltas.append(a-b if a is not None and b is not None else None)
            tables["paired_rho_deltas"].append({"model_id": model, "reference_model": reference, "mode": mode, **_summary(deltas, "delta_rho")})
    return {"schema_version": 1, "bundle_manifest_sha256": bundle_hash, "reference_model": reference,
            "contract": {"analysis": shared_contract, "seed_ids": list(SEEDS), "mean_and_sample_sd": True,
                         "summary_requires_five_defined_values": True, "rho_delta": "paired by seed vs reference",
                         "boundary_is_exploratory_data_selected": True, "boundary_min_side_fraction": 0.05,
                         "bootstrap": False, "rho_mcse": False, "rendering": False}, "tables": tables, "plot_data": plot}


def score_contextual_dimsim_batch(bundle_path: str | Path, run_dirs: dict[str, str | Path],
                                  representation_dirs: dict[str, dict[int, str | Path]],
                                  analysis_config: dict | None = None) -> dict:
    """Score explicit model run directories and explicit model/seed representation paths."""
    if not isinstance(run_dirs, dict) or not run_dirs or not isinstance(representation_dirs, dict):
        raise ValueError("explicit run and representation mappings are required")
    runs, reps = {}, {}
    for source, target in ((run_dirs, runs), (representation_dirs, reps)):
        for key, value in source.items():
            canonical = canonical_model_id(key)
            if canonical in target: raise ValueError("model aliases collide after canonicalization")
            target[canonical] = value
    if set(runs) != set(reps): raise ValueError("model inventories differ")
    bundle = load_evaluation_bundle(bundle_path)
    bundle_sha = hashlib.sha256((Path(bundle_path) / "manifest.json").read_bytes()).hexdigest()
    source_hashes = bundle["manifest"].get("source_experiment", {}).get("hashes")
    scored = []
    for model in sorted(runs):
        run = discover_seed_runs(runs[model], model)
        seed_paths = reps[model]
        if not isinstance(seed_paths, dict) or any(type(s) is not int for s in seed_paths) or set(seed_paths) != set(SEEDS):
            raise ValueError(f"representations must map seeds 0 through 4 for {model}")
        for record in run["seeds"]:
            seed = record["seed"]
            loaded = load_contextual_representations(seed_paths[seed])
            p = loaded["manifest"]["provenance"]
            if p["model_id"] != model or p["seed"] != seed or p["checkpoint_sha256"] != record["checkpoint_sha256"] or p["experiment_hashes"] != run["experiment_hashes"]:
                raise ValueError(f"representation/run provenance mismatch for {model} seed {seed}")
            if any(run["experiment_hashes"].get(k) != v for k, v in source_hashes.items()):
                raise ValueError("bundle and training run source hashes differ")
            score = score_contextual_dimsim(bundle, loaded["occurrence_ids"], loaded["hidden"], p, bundle_sha, analysis_config)
            scored.append({"model_id": model, "seed": seed, "run_id": run["run_id"],
                           "run_manifest_sha256": run["manifest_sha256"], "checkpoint_sha256": record["checkpoint_sha256"], "score": score})
    return {"schema_version": 1, "bundle_manifest_sha256": bundle_sha, "results": scored}
