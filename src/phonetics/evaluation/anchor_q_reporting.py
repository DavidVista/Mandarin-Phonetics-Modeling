"""Pure, full-precision tables and plot data for complete anchor-Q batches.

The report preserves model/seed identity. MC SE describes fixed-draw Monte Carlo
variation; sample SD describes variation across the five training seeds. They are
reported separately and are never combined into confidence intervals.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

from ..objectives import canonical_model_id
from .anchor_sampling import EDGES

SEEDS = (0, 1, 2, 3, 4)
COMPONENTS = ("initial", "final", "tone")
METRICS = (*COMPONENTS, "overall")
EDGE_ORDER = tuple(f"{component}:{low}->{high}" for component, low, high in EDGES)


def _number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _mean_se(draws: list[np.ndarray]) -> tuple[float, float | None]:
    """MC SE of the seed-mean estimator, retaining corresponding draw pairing."""
    pooled = np.mean(np.stack(draws, axis=0), axis=0)
    if pooled.size < 2:
        return float(np.mean(pooled)), None
    return float(np.mean(pooled)), float(np.std(pooled, ddof=1) / math.sqrt(pooled.size))


def _edge_ids(draws: list[dict[str, Any]], edges: list[str]) -> tuple[list[tuple[str, str, str]], dict[str, list[float]]]:
    identities = []
    values = {edge: [] for edge in edges}
    for row in draws:
        if not isinstance(row, dict):
            raise ValueError("draw score rows must be objects")
        draw_id, edge_id, triplet_id = (row.get(key) for key in ("draw_id", "edge_id", "triplet_id"))
        if not all(isinstance(x, str) and x for x in (draw_id, edge_id, triplet_id)):
            raise ValueError("draw identities must be non-empty strings")
        if edge_id not in values:
            raise ValueError(f"unexpected edge identity: {edge_id}")
        score = row.get("score")
        if not _number(score) or float(score) not in (0.0, 0.5, 1.0):
            raise ValueError("draw score must be numeric 0, 0.5, or 1")
        identities.append((draw_id, edge_id, triplet_id))
        values[edge_id].append(float(score))
    if len({item[0] for item in identities}) != len(identities):
        raise ValueError("draw IDs must be unique within each seed")
    expected_edge_sequence = [edge for edge in edges for _ in values[edge]]
    if [item[1] for item in identities] != expected_edge_sequence:
        raise ValueError("draw rows must follow canonical edge blocks")
    return identities, values


def build_anchor_q_report(batch_result: dict[str, Any], reference_model: str = "syllable") -> dict[str, Any]:
    """Validate paired five-seed batch scores and return tidy JSON-compatible data.

    ``tables`` contains per-model/per-seed edges and metrics, per-model seed
    summaries, per-model global summaries, paired deltas, and normalized shares.
    ``plot_data`` contains edge means with their five seed points and the two share
    series. All values retain full numeric precision. Missing one-draw MC SE is
    represented by ``None`` and an explicit status.
    """
    if not isinstance(batch_result, dict) or type(batch_result.get("schema_version")) is not int or batch_result["schema_version"] != 1:
        raise ValueError("batch schema_version must be integer 1")
    bundle_hash = batch_result.get("bundle_manifest_sha256")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64 or any(c not in "0123456789abcdef" for c in bundle_hash):
        raise ValueError("invalid batch bundle manifest SHA-256")
    try:
        reference = canonical_model_id(reference_model)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid reference model") from error
    records = batch_result.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError("batch results are required")
    models = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("model_id"), str):
            raise ValueError("each batch record requires a model ID")
        try:
            model = canonical_model_id(record["model_id"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid model ID") from error
        if model != record["model_id"]:
            raise ValueError("model IDs must be canonical")
        models.add(model)
    if reference not in models:
        raise ValueError("reference model is absent from batch")
    model_order = sorted(models)
    grouped: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in model_order}
    for record in records:
        model, seed = record["model_id"], record.get("seed")
        if type(seed) is not int or seed not in SEEDS or seed in grouped[model]:
            raise ValueError(f"invalid or duplicate seed for {model}")
        score = record.get("score")
        if not isinstance(score, dict):
            raise ValueError("score must be an object")
        if score.get("model_id") != model or type(score.get("seed")) is not int or score["seed"] != seed:
            raise ValueError("internal model/seed metadata mismatch")
        if score.get("bundle_manifest_sha256") != bundle_hash:
            raise ValueError("bundle hash mismatch")
        provenance = score.get("provenance")
        if (not isinstance(provenance, dict) or provenance.get("model_id") != model
                or type(provenance.get("seed")) is not int or provenance["seed"] != seed
                or provenance.get("bundle_manifest_sha256") != bundle_hash):
            raise ValueError("score provenance model/seed/bundle metadata mismatch")
        tolerance = score.get("tie_tolerance")
        if not _number(tolerance) or tolerance < 0:
            raise ValueError("invalid tie tolerance")
        grouped[model][seed] = score
    if any(set(scores) != set(SEEDS) for scores in grouped.values()):
        raise ValueError("every model requires seeds 0 through 4 exactly once")

    first_score = grouped[model_order[0]][SEEDS[0]]
    tolerance = float(first_score["tie_tolerance"])
    edge_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    all_draw_values: dict[str, dict[int, dict[str, np.ndarray]]] = {model: {} for model in model_order}
    identity_reference = None
    draw_count = None
    coverage_reference = None
    for model in model_order:
        for seed in SEEDS:
            score = grouped[model][seed]
            if float(score["tie_tolerance"]) != tolerance:
                raise ValueError("tie tolerance mismatch")
            edge_summary = score.get("edges")
            if not isinstance(edge_summary, dict) or set(edge_summary) != set(EDGE_ORDER):
                raise ValueError("score edges must contain the canonical twelve-edge inventory")
            draws = score.get("draw_scores")
            if not isinstance(draws, list) or not draws:
                raise ValueError("draw_scores must be a non-empty list")
            identities, edge_values = _edge_ids(draws, list(EDGE_ORDER))
            expected_draw_ids = [f"anchor-q:v1:{edge}:{index:06d}"
                                 for edge in EDGE_ORDER for index in range(len(edge_values[edge]))]
            if [identity[0] for identity in identities] != expected_draw_ids:
                raise ValueError("draw IDs do not match canonical edge/index order")
            if identity_reference is None:
                identity_reference = identities
            elif identities != identity_reference:
                raise ValueError("draw identities differ across models or seeds")
            sizes = {edge: len(edge_values[edge]) for edge in EDGE_ORDER}
            if len(set(sizes.values())) != 1:
                raise ValueError("all edge draw counts must match")
            if draw_count is None:
                draw_count = next(iter(sizes.values()))
            if any(count != draw_count for count in sizes.values()):
                raise ValueError("edge draw counts differ across models or seeds")
            coverage = score.get("coverage")
            if not isinstance(coverage, dict) or not isinstance(coverage.get("edges"), dict) or set(coverage["edges"]) != set(EDGE_ORDER):
                raise ValueError("per-edge coverage diagnostics are required")
            if coverage_reference is None:
                coverage_reference = coverage
            elif coverage != coverage_reference:
                raise ValueError("coverage diagnostics differ across models or seeds")
            arrays = {edge: np.asarray(edge_values[edge], dtype=np.float64) for edge in EDGE_ORDER}
            all_draw_values[model][seed] = arrays
            q = {}
            mcse = {}
            for edge in EDGE_ORDER:
                vals = arrays[edge]
                q[edge] = float(np.mean(vals))
                mcse[edge] = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else None
                declared = edge_summary[edge]
                if not isinstance(declared, dict) or type(declared.get("draw_count")) is not int or declared["draw_count"] != vals.size:
                    raise ValueError("declared edge draw_count differs from actual draws")
                cov = coverage["edges"][edge]
                edge_rows.append({"model_id": model, "seed": seed, "edge_id": edge, "q": q[edge],
                                  "mc_se": mcse[edge], "mc_se_status": "ok" if mcse[edge] is not None else "not_available_single_draw",
                                  "draw_count": int(vals.size), "coverage": cov})
            for component in COMPONENTS:
                edges = [edge for edge in EDGE_ORDER if edge.startswith(component + ":")]
                metric_rows.append({"model_id": model, "seed": seed, "metric": component,
                                    "q": float(np.mean([q[e] for e in edges])),
                                    "mc_se": float(math.sqrt(sum(mcse[e] ** 2 for e in edges)) / len(edges)) if all(mcse[e] is not None for e in edges) else None,
                                    "mc_se_status": "ok" if all(mcse[e] is not None for e in edges) else "not_available_single_draw"})
            metric_rows.append({"model_id": model, "seed": seed, "metric": "overall",
                                "q": float(np.mean([q[e] for e in EDGE_ORDER])),
                                "mc_se": float(math.sqrt(sum(mcse[e] ** 2 for e in EDGE_ORDER)) / len(EDGE_ORDER)) if all(mcse[e] is not None for e in EDGE_ORDER) else None,
                                "mc_se_status": "ok" if all(mcse[e] is not None for e in EDGE_ORDER) else "not_available_single_draw"})

    edge_summary_rows = []
    for model in model_order:
        for edge in EDGE_ORDER:
            seed_values = [float(np.mean(all_draw_values[model][seed][edge])) for seed in SEEDS]
            mean_q, mean_mcse = _mean_se([all_draw_values[model][seed][edge] for seed in SEEDS])
            edge_summary_rows.append({"model_id": model, "edge_id": edge, "mean": mean_q,
                                      "sample_sd": _sample_sd(seed_values), "mc_se": mean_mcse,
                                      "mc_se_status": "ok" if mean_mcse is not None else "not_available_single_draw",
                                      "seed_values": [{"seed": seed, "q": val} for seed, val in zip(SEEDS, seed_values)]})
    metric_summary_rows = []
    global_summary_rows = []
    for model in model_order:
        for metric in METRICS:
            edge_subset = list(EDGE_ORDER) if metric == "overall" else [e for e in EDGE_ORDER if e.startswith(metric + ":")]
            seed_values = []
            edge_mean_ses = []
            for seed in SEEDS:
                edge_means = [float(np.mean(all_draw_values[model][seed][edge])) for edge in edge_subset]
                seed_values.append(float(np.mean(edge_means)))
            mean_q = float(np.mean(seed_values))
            for edge in edge_subset:
                _, edge_mean_se = _mean_se([all_draw_values[model][seed][edge] for seed in SEEDS])
                edge_mean_ses.append(edge_mean_se)
            mean_mcse = (float(math.sqrt(sum(value ** 2 for value in edge_mean_ses)) / len(edge_subset))
                         if all(value is not None for value in edge_mean_ses) else None)
            row = {"model_id": model, "metric": metric, "mean": mean_q,
                   "sample_sd": _sample_sd(seed_values), "mc_se": mean_mcse,
                   "mc_se_status": "ok" if mean_mcse is not None else "not_available_single_draw",
                   "seed_values": [{"seed": seed, "q": val} for seed, val in zip(SEEDS, seed_values)]}
            metric_summary_rows.append(row)
            global_summary_rows.append(dict(row))

    paired_rows = []
    for model in model_order:
        if model == reference:
            continue
        for metric in METRICS:
            edge_subset = list(EDGE_ORDER) if metric == "overall" else [e for e in EDGE_ORDER if e.startswith(metric + ":")]
            seed_deltas = []
            mean_draw_differences = []
            for seed in SEEDS:
                edge_diff_means = []
                for edge in edge_subset:
                    diffs = all_draw_values[model][seed][edge] - all_draw_values[reference][seed][edge]
                    edge_diff_means.append(float(np.mean(diffs)))
                seed_deltas.append(float(np.mean(edge_diff_means)))
            for edge in edge_subset:
                across_seeds = np.mean(np.stack([
                    all_draw_values[model][seed][edge] - all_draw_values[reference][seed][edge]
                    for seed in SEEDS]), axis=0)
                mean_draw_differences.append(across_seeds)
            edge_mcse = [float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else None
                         for values in mean_draw_differences]
            paired_mcse = float(math.sqrt(sum(value ** 2 for value in edge_mcse) / len(edge_mcse) ** 2)) if all(value is not None for value in edge_mcse) else None
            paired_rows.append({"model_id": model, "reference_model": reference, "metric": metric,
                                "mean_delta_q": float(np.mean(seed_deltas)), "sample_sd_seed_delta": _sample_sd(seed_deltas),
                                "mc_se": paired_mcse, "mc_se_status": "ok" if paired_mcse is not None else "not_available_single_draw",
                                "seed_deltas": [{"seed": seed, "delta_q": value} for seed, value in zip(SEEDS, seed_deltas)]})

    share_rows = []
    component_share_rows = []
    for model in model_order:
        means = {row["edge_id"]: row["mean"] for row in edge_summary_rows if row["model_id"] == model}
        total = sum(means.values())
        for edge in EDGE_ORDER:
            share_rows.append({"model_id": model, "edge_id": edge,
                               "share": means[edge] / total if total else None,
                               "status": "ok" if total else "undefined_zero_total"})
        comp_means = {
            component: float(np.mean([means[e] for e in EDGE_ORDER if e.startswith(component + ":")]))
            for component in COMPONENTS
        }
        comp_total = sum(comp_means.values())
        for component in COMPONENTS:
            component_share_rows.append({"model_id": model, "component": component,
                                         "share": comp_means[component] / comp_total if comp_total else None,
                                         "status": "ok" if comp_total else "undefined_zero_total"})

    plot_edges = [{"model_id": model, "edge_id": edge,
                   "mean": next(row["mean"] for row in edge_summary_rows if row["model_id"] == model and row["edge_id"] == edge),
                   "seed_points": [{"seed": seed, "q": float(np.mean(all_draw_values[model][seed][edge]))} for seed in SEEDS]}
                  for model in model_order for edge in EDGE_ORDER]
    return {"schema_version": 1,
            "contract": {"seeds": list(SEEDS), "edge_order": list(EDGE_ORDER), "metrics": list(METRICS),
                         "reference_model": reference, "bundle_manifest_sha256": bundle_hash,
                         "tie_tolerance": tolerance, "draws_per_edge": draw_count,
                         "full_precision": True, "seed_summary": "mean and sample SD (ddof=1) across five seeds",
                         "mc_uncertainty": "MC SE from draw scores averaged across seeds before sample SD / sqrt(draw count); independent-edge aggregation",
                         "paired_uncertainty": "paired draw differences, averaged across seeds before MC SE; no confidence intervals",
                         "tables": ["model_seed_edges", "model_seed_metrics", "model_edge_seed_summary", "model_metric_seed_summary", "global_metric_summary", "paired_deltas", "transition_shares", "component_shares"],
                         "plot_data": ["edges with mean and five seed points", "transition shares", "component shares"]},
            "tables": {"model_seed_edges": edge_rows, "model_seed_metrics": metric_rows,
                       "model_edge_seed_summary": edge_summary_rows, "model_metric_seed_summary": metric_summary_rows,
                       "global_metric_summary": global_summary_rows, "paired_deltas": paired_rows,
                       "transition_shares": share_rows, "component_shares": component_share_rows},
            "plot_data": {"edges": plot_edges, "transition_shares": share_rows, "component_shares": component_share_rows}}
