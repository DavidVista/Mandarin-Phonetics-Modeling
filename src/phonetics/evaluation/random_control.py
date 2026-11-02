"""Matched iid Gaussian occurrence controls for contextual evaluation."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_q import score_anchor_q
from .anchor_sampling import EDGES
from .bundle import load_evaluation_bundle
from .contextual_dimsim import score_contextual_dimsim
from .representations import load_contextual_representations, validate_matrix
from ..objectives import canonical_model_id

RANDOM_SEEDS = (0, 1, 2, 3, 4)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_contextual_dimension(model_runs: dict[str, dict[int, str | Path]], *,
                                 bundle_manifest_sha256: str, bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate all learned representation runs and pin the shared h dimension."""
    if not isinstance(model_runs, dict) or not model_runs:
        raise ValueError("explicit learned model/seed representation paths are required")
    dimensions, inventory, pins = set(), None, {}
    expected_inventory = tuple(sorted(row["occurrence_id"] for row in bundle["occurrences"]))
    if bundle.get("manifest", {}).get("schema_version") != 3 or not _is_sha256(bundle_manifest_sha256):
        raise ValueError("dimension resolution requires a verified schema-3 bundle and manifest hash")
    for model_id, seed_runs in sorted(model_runs.items()):
        try:
            if canonical_model_id(model_id) != model_id:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"learned model ID is not canonical: {model_id}") from error
        if (not isinstance(seed_runs, dict) or any(type(seed) is not int for seed in seed_runs)
                or set(seed_runs) != set(RANDOM_SEEDS)):
            raise ValueError(f"{model_id}: exactly contextual seeds 0..4 are required")
        pins[model_id] = {}
        for seed in RANDOM_SEEDS:
            loaded = load_contextual_representations(seed_runs[seed])
            manifest = loaded["manifest"]
            provenance = manifest["provenance"]
            if provenance["model_id"] != model_id or provenance["seed"] != seed:
                raise ValueError(f"learned representation identity mismatch for {model_id}/{seed}")
            if provenance["bundle_manifest_sha256"] != bundle_manifest_sha256:
                raise ValueError("learned representations do not share the requested bundle")
            ids = tuple(loaded["occurrence_ids"])
            if inventory is None:
                inventory = ids
            elif ids != inventory:
                raise ValueError("learned representations have mismatched occurrence inventories")
            dimensions.add(int(manifest["dimension"]))
            pins[model_id][str(seed)] = {
                "path": str(Path(seed_runs[seed]).resolve()),
                "manifest_sha256": _sha(Path(seed_runs[seed]) / "manifest.json"),
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "dimension": int(manifest["dimension"]),
                "occurrence_ids": list(ids),
            }
    if inventory != expected_inventory:
        raise ValueError("learned representations do not match the requested bundle occurrence inventory")
    if len(dimensions) != 1:
        raise ValueError("learned contextual dimensions differ across models or seeds")
    return {"schema_version": 1, "dimension": dimensions.pop(), "models": pins,
            "bundle_manifest_sha256": bundle_manifest_sha256, "occurrence_ids": list(expected_inventory)}


def generate_random_control(occurrence_ids: list[str], *, dimension: int,
                            random_seeds: tuple[int, ...] | list[int] = RANDOM_SEEDS) -> dict[str, Any]:
    """Generate five deterministic standard-normal matrices; retain arrays outside reports."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a strict positive integer")
    if (type(random_seeds) not in (tuple, list) or any(type(seed) is not int for seed in random_seeds)
            or tuple(random_seeds) != RANDOM_SEEDS):
        raise ValueError("random seeds are fixed to integer seeds 0..4")
    if (not isinstance(occurrence_ids, list) or not occurrence_ids
            or any(not isinstance(x, str) or not x for x in occurrence_ids)
            or occurrence_ids != sorted(occurrence_ids) or len(set(occurrence_ids)) != len(occurrence_ids)):
        raise ValueError("occurrence IDs must be nonempty, unique, and sorted")
    arrays = {}
    for seed in RANDOM_SEEDS:
        rng = np.random.Generator(np.random.PCG64(seed))
        arrays[seed] = validate_matrix(occurrence_ids,
            rng.standard_normal((len(occurrence_ids), dimension)).astype(np.float32))
    return {"occurrence_ids": list(occurrence_ids), "matrices": arrays}


def evaluate_random_control(bundle: dict[str, Any], generated: dict[str, Any], *,
                            bundle_manifest_sha256: str,
                            dimension_resolution: dict[str, Any], source_sha256: str,
                            analysis_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score the same generated vectors on Q and DIMSIM, returning report and raw arrays separately."""
    ids, matrices = generated["occurrence_ids"], generated["matrices"]
    if not _is_sha256(source_sha256):
        raise ValueError("explicit build source SHA-256 is required")
    _validate_dimension_resolution(dimension_resolution, bundle, bundle_manifest_sha256, ids)
    if (not isinstance(matrices, dict) or any(type(seed) is not int for seed in matrices)
            or set(matrices) != set(RANDOM_SEEDS)):
        raise ValueError("generated control must contain exactly random seeds 0..4")
    if bundle["manifest"].get("schema_version") != 3:
        raise ValueError("random control requires a verified schema-3 bundle")
    records = []
    for random_seed in RANDOM_SEEDS:
        matrix = np.asarray(matrices[random_seed])
        if (matrix.ndim != 2 or matrix.shape != (len(ids), dimension_resolution["dimension"])
                or matrix.dtype != np.dtype("float32") or not np.isfinite(matrix).all()
                or np.any(np.max(np.abs(matrix.astype(np.float64)), axis=1) == 0)):
            raise ValueError(f"random matrix {random_seed} does not match the resolved dimension or finite float32 contract")
        provenance = {"model_id": "random-control", "seed": random_seed,
                      "seed_kind": "random", "adapter": "iid_standard_normal",
                      "rng": "PCG64", "rng_distribution": "standard_normal",
                      "numpy_version": np.__version__, "source_sha256": source_sha256,
                      "dimension": int(matrix.shape[1]), "dtype": str(matrix.dtype),
                      "bundle_manifest_sha256": bundle_manifest_sha256,
                      "dimension_resolution": dimension_resolution}
        q = score_anchor_q(bundle, ids, matrix, provenance,
                           bundle_manifest_sha256=bundle_manifest_sha256,
                           tie_tolerance=bundle["config"]["scoring"]["anchor_q"]["tie_tolerance"])
        d = score_contextual_dimsim(bundle, ids, matrix, provenance,
                                    bundle_manifest_sha256, analysis_config)
        records.append({"random_seed": random_seed, "seed_kind": "random", "provenance": provenance,
                        "anchor_q": q, "dimsim": d})
    return {"report": summarize_random_control(records, bundle_manifest_sha256), "matrices": matrices,
            "occurrence_ids": ids}


def _mean_sd(values: list[float]) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    return {"mean": float(vals.mean()), "sample_sd": float(vals.std(ddof=1)), "count": int(len(vals))}


def summarize_random_control(records: list[dict[str, Any]], bundle_manifest_sha256: str) -> dict[str, Any]:
    if not _is_sha256(bundle_manifest_sha256) or type(records) is not list or len(records) != 5:
        raise ValueError("random control summary requires five records and a valid bundle hash")
    _validate_summary_records(records, bundle_manifest_sha256)
    edges = [f"{component}:{low}->{high}" for component, low, high in EDGES]
    q_summary = {"edges": {}, "components": {}, "overall": {}}
    per_seed_edge_means: dict[str, list[float]] = {}
    for edge in edges:
        seed_values = _seed_edge_means(records, edge)
        per_seed_edge_means[edge] = seed_values
        vals = _matched_q_values(records, edge)
        q_summary["edges"][edge] = {**_mean_sd(seed_values), **_mcse(vals)}
    for component in ("initial", "final", "tone"):
        component_edges = [e for e in edges if e.split(":", 1)[0] == component]
        edge_mc = [q_summary["edges"][e]["mc_se_seed_mean"] for e in component_edges]
        seed_values = [float(np.mean([per_seed_edge_means[e][i] for e in component_edges])) for i in range(5)]
        mcse = _combine_edge_mcse(edge_mc)
        q_summary["components"][component] = {**_mean_sd(seed_values), "mc_se_seed_mean": mcse,
            "mc_se_status": "ok" if mcse is not None else "not_available_single_draw"}
    edge_mc = [q_summary["edges"][e]["mc_se_seed_mean"] for e in edges]
    seed_values = [float(np.mean([per_seed_edge_means[e][i] for e in edges])) for i in range(5)]
    mcse = _combine_edge_mcse(edge_mc)
    q_summary["overall"] = {**_mean_sd(seed_values), "mc_se_seed_mean": mcse,
                             "mc_se_status": "ok" if mcse is not None else "not_available_single_draw"}
    dim_summary = {}
    for mode in records[0]["dimsim"]["modes"]:
        rhos = [_validated_rho(r["dimsim"]["modes"][mode]["summary"]["rho"]) for r in records]
        defined = [x for x in rhos if x is not None]
        complete = len(defined) == 5
        dim_summary[mode] = {"rho_mean": float(np.mean(defined)) if complete else None,
                             "rho_sample_sd": float(np.std(defined, ddof=1)) if complete else None,
                             "defined_seed_count": len(defined), "status": "ok" if complete else "requires_all_five_defined",
                             "random_seeds": list(RANDOM_SEEDS),
                             "per_seed_rho": [{"random_seed": r["random_seed"], "rho": r["dimsim"]["modes"][mode]["summary"]["rho"]["rho"],
                                               "undefined_reason": r["dimsim"]["modes"][mode]["summary"]["rho"].get("undefined_reason")}
                                              for r in records]}
    return {"schema_version": 1, "model_id": "random-control", "adapter": "iid_standard_normal",
            "seed_kind": "random", "random_seeds": list(RANDOM_SEEDS),
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "interpretation": "Variation across these controls reflects random-vector variation, distinct from training-seed variation.",
            "anchor_q": q_summary, "dimsim": dim_summary,
            "per_seed": records}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_dimension_resolution(resolution: Any, bundle: dict[str, Any], bundle_hash: str,
                                   occurrence_ids: list[str]) -> None:
    if (not isinstance(resolution, dict) or resolution.get("schema_version") != 1
            or type(resolution.get("dimension")) is not int or resolution["dimension"] <= 0
            or resolution.get("bundle_manifest_sha256") != bundle_hash
            or resolution.get("occurrence_ids") != occurrence_ids):
        raise ValueError("validated dimension resolution does not match bundle and occurrence inventory")
    models = resolution.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("dimension resolution lacks learned model pins")
    for model, seeds in models.items():
        try:
            if canonical_model_id(model) != model:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError("dimension resolution contains noncanonical model identity") from error
        if not isinstance(seeds, dict) or set(seeds) != {str(x) for x in RANDOM_SEEDS}:
            raise ValueError("dimension resolution must pin exact learned seeds 0..4")
        for seed, pin in seeds.items():
            if (not isinstance(pin, dict) or not _is_sha256(pin.get("manifest_sha256"))
                    or not _is_sha256(pin.get("checkpoint_sha256"))
                    or pin.get("dimension") != resolution["dimension"]
                    or pin.get("occurrence_ids") != occurrence_ids):
                raise ValueError(f"invalid learned dimension pin {model}/{seed}")


def _validate_summary_records(records: list[dict[str, Any]], bundle_hash: str) -> None:
    reference_provenance = None
    for expected, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get("random_seed")) is not int
                or record["random_seed"] != expected or record.get("seed_kind") != "random"):
            raise ValueError("random control summary requires exact ordered integer seeds 0..4")
        q, d = record.get("anchor_q"), record.get("dimsim")
        if (not isinstance(q, dict) or not isinstance(d, dict) or q.get("bundle_manifest_sha256") != bundle_hash
                or d.get("bundle_manifest_sha256") != bundle_hash or q.get("model_id") != "random-control"
                or q.get("seed") != expected or d.get("model_id") != "random-control" or d.get("seed") != expected):
            raise ValueError("random control score identity or bundle hash mismatch")
        provenance = record.get("provenance")
        required = {"adapter": "iid_standard_normal", "rng": "PCG64", "rng_distribution": "standard_normal",
                    "seed_kind": "random", "model_id": "random-control", "seed": expected,
                    "bundle_manifest_sha256": bundle_hash}
        if (not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in required.items())
                or not _is_sha256(provenance.get("source_sha256"))):
            raise ValueError("random control provenance or RNG contract is invalid")
        invariant = {key: value for key, value in provenance.items() if key != "seed"}
        if reference_provenance is None:
            reference_provenance = invariant
        elif invariant != reference_provenance:
            raise ValueError("random control provenance differs across seeds")
    base_q = records[0]["anchor_q"]["draw_scores"]
    edges = [f"{component}:{low}->{high}" for component, low, high in EDGES]
    if not isinstance(records[0]["anchor_q"].get("edges"), dict) or set(records[0]["anchor_q"]["edges"]) != set(edges):
        raise ValueError("random control Q edge inventory/order mismatch")
    base_triplets = [(r["draw_id"], r["triplet_id"], r["edge_id"]) for r in base_q]
    _validate_q_draws(base_q, edges)
    base_dims = records[0]["dimsim"]
    base_dim_rows = {}
    for mode, result in base_dims["modes"].items():
        base_dim_rows[mode] = _dimsim_contract_rows(result["rows"])
    for record in records[1:]:
        q, d = record["anchor_q"], record["dimsim"]
        if not isinstance(q.get("edges"), dict) or set(q["edges"]) != set(edges):
            raise ValueError("random control Q edge inventory/order mismatch")
        if [(r["draw_id"], r["triplet_id"], r["edge_id"]) for r in q["draw_scores"]] != base_triplets:
            raise ValueError("random control Q triplet identity mismatch")
        _validate_q_draws(q["draw_scores"], edges)
        if q["tie_tolerance"] != records[0]["anchor_q"]["tie_tolerance"]:
            raise ValueError("random control Q scoring contract mismatch")
        for mode in base_dims["modes"]:
            left, right = base_dims["modes"][mode], d["modes"].get(mode)
            if (right is None or left["summary"]["draw_count"] != right["summary"]["draw_count"]
                    or [(x["draw_id"], x["pair_id"], x["left_id"], x["right_id"]) for x in left["rows"]]
                    != [(x["draw_id"], x["pair_id"], x["left_id"], x["right_id"]) for x in right["rows"]]
                    or _dimsim_contract_rows(right["rows"]) != base_dim_rows[mode]
                    or d["contract"] != base_dims["contract"]):
                raise ValueError("random control DIMSIM pair identity or analysis contract mismatch")


def _validate_q_draws(rows: list[dict[str, Any]], edges: list[str]) -> None:
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("random control Q draw table is invalid")
    edge_count = len(rows) // len(edges)
    if edge_count <= 0 or len(rows) != edge_count * len(edges):
        raise ValueError("random control Q draw coverage is incomplete")
    for index, row in enumerate(rows):
        edge_index, draw_index = divmod(index, edge_count)
        score = row.get("score")
        if (row.get("edge_id") != edges[edge_index] or type(score) not in (int, float)
                or not math.isfinite(score) or score not in (0, 0.5, 1)
                or not isinstance(row.get("draw_id"), str) or not row["draw_id"]
                or not isinstance(row.get("triplet_id"), str) or not row["triplet_id"]):
            raise ValueError("random control Q draw values or canonical edge order are invalid")
        if row["draw_id"] != f"anchor-q:v1:{edges[edge_index]}:{draw_index:06d}":
            raise ValueError("random control Q draw ID/index order is invalid")


def _matched_q_values(records: list[dict[str, Any]], edge: str) -> list[float]:
    by_seed = [{row["draw_id"]: float(row["score"]) for row in record["anchor_q"]["draw_scores"] if row["edge_id"] == edge}
               for record in records]
    if any(set(values) != set(by_seed[0]) for values in by_seed[1:]):
        raise ValueError("matched Q draw identities differ across random seeds")
    ids = list(by_seed[0])
    return [float(np.mean([values[draw_id] for values in by_seed])) for draw_id in ids]


def _seed_edge_means(records: list[dict[str, Any]], edge: str) -> list[float]:
    result = []
    for record in records:
        scores = [float(row["score"]) for row in record["anchor_q"]["draw_scores"] if row["edge_id"] == edge]
        if not scores:
            raise ValueError(f"random control Q edge has no draws: {edge}")
        result.append(float(np.mean(scores)))
    return result


def _validated_rho(value: Any) -> float | None:
    if not isinstance(value, dict):
        raise ValueError("random control DIMSIM rho record is invalid")
    rho = value.get("rho")
    if rho is None:
        reason = value.get("undefined_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("undefined random control rho requires a reason")
        return None
    if type(rho) not in (int, float) or not math.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError("random control DIMSIM rho must be finite and within [-1,1]")
    return float(rho)


def _dimsim_contract_rows(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    if not isinstance(rows, list):
        raise ValueError("DIMSIM rows must be a list")
    result = []
    for row in rows:
        distance = row.get("dimsim_distance")
        if distance is not None and (type(distance) not in (int, float) or not math.isfinite(distance)):
            raise ValueError("DIMSIM distance metadata must be finite or null")
        for key in ("draw_id", "pair_id", "left_id", "right_id", "analysis_status", "distance_bin"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"DIMSIM row lacks valid {key}")
        if row.get("distance_status") != "ok" and distance is not None:
            raise ValueError("invalid DIMSIM metric must not carry a numeric distance")
        result.append((row["draw_id"], row["pair_id"], row["left_id"], row["right_id"], distance,
                       row.get("distance_status"), row.get("overlap_mask"),
                       row.get("no_zero_initial_diagnostic"), row["analysis_status"], row["distance_bin"]))
    return result


def _mcse(values: list[float]) -> dict[str, Any]:
    if len(values) <= 1:
        return {"mc_se_seed_mean": None, "mc_se_status": "not_available_single_draw", "draw_count": len(values)}
    return {"mc_se_seed_mean": float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values))),
            "mc_se_status": "ok", "draw_count": len(values)}


def _combine_edge_mcse(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(math.sqrt(sum(float(value) ** 2 for value in values)) / len(values))
