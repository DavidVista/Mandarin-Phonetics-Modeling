"""Pure coordinator for the independent contextual Anchor-Q and DIMSIM reports."""
from __future__ import annotations

import json
import re
from typing import Any

from ..objectives import canonical_model_id
from ..research.artifacts import safe_run_id
from .anchor_q_reporting import build_anchor_q_report
from .dimsim_reporting import build_dimsim_report

SEEDS = (0, 1, 2, 3, 4)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_HASH_NAMES = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"}
_FIGURES = (
    ("Anchor-Q transitions", "contextual-anchor-q", {"mean_and_seed_points": "plot_data.edges"}),
    ("Transition shares", "contextual-anchor-q", {"shares": "plot_data.transition_shares"}),
    ("Component shares", "contextual-anchor-q", {"shares": "plot_data.component_shares"}),
    ("DIMSIM geometry", "contextual-dimsim", {"curves": "plot_data.curves", "boundaries": "plot_data.boundaries", "counts": "plot_data.mode_counts"}),
    ("DIMSIM correlation", "contextual-dimsim", {"model_rho": "tables.model_rho", "seed_rho": "tables.seed_rho"}),
)


def _batch_identity(batch: dict[str, Any], label: str) -> tuple[str, tuple[str, ...], dict[tuple[str, int], dict]]:
    if not isinstance(batch, dict) or type(batch.get("schema_version")) is not int or batch["schema_version"] != 1:
        raise ValueError(f"{label} batch schema_version must be integer 1")
    bundle_hash = batch.get("bundle_manifest_sha256")
    if not isinstance(bundle_hash, str) or not _SHA.fullmatch(bundle_hash):
        raise ValueError(f"{label} batch bundle hash is invalid")
    records = batch.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} batch results are required")
    grouped = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{label} record must be an object")
        model, seed = row.get("model_id"), row.get("seed")
        try:
            canonical = canonical_model_id(model)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} model identity is invalid") from exc
        if model != canonical or type(seed) is not int or seed not in SEEDS:
            raise ValueError(f"{label} model/seed identity is invalid")
        key = (model, seed)
        if key in grouped:
            raise ValueError(f"{label} has duplicate model/seed")
        for field in ("run_id", "run_manifest_sha256", "checkpoint_sha256"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{label} {field} is required for {model} seed {seed}")
        for field in ("run_manifest_sha256", "checkpoint_sha256"):
            if not _SHA.fullmatch(row[field]):
                raise ValueError(f"{label} {field} is invalid")
        hashes = row.get("experiment_hashes")
        if hashes is not None and (not isinstance(hashes, dict) or set(hashes) != _EXPERIMENT_HASH_NAMES
                                   or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in hashes.values())):
            raise ValueError(f"{label} experiment_hashes inventory is invalid")
        grouped[key] = row
    models = tuple(sorted({model for model, _ in grouped}))
    expected = {(model, seed) for model in models for seed in SEEDS}
    if set(grouped) != expected:
        raise ValueError(f"{label} must contain all five seeds for every model")
    for model in models:
        model_rows = [grouped[(model, seed)] for seed in SEEDS]
        for field in ("run_id", "run_manifest_sha256", "experiment_hashes"):
            available = [row.get(field) for row in model_rows if row.get(field) is not None]
            if available and any(value != available[0] for value in available[1:]):
                raise ValueError(f"{label} {field} differs across seeds for {model}")
    return bundle_hash, models, grouped


def _semantic(report: dict, path: str) -> Any:
    value: Any = report
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"figure data mapping does not resolve: {path}")
        value = value[key]
    return value


def _json_finite(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON-compatible data") from exc


def build_evaluation_report(q_batch: dict, dimsim_batch: dict, reference_model: str = "syllable",
                            run_ids: dict[str, str] | None = None) -> dict:
    """Build a validated JSON contract and artifact plan without writing files."""
    q_bundle, q_models, q_rows = _batch_identity(q_batch, "Anchor-Q")
    d_bundle, d_models, d_rows = _batch_identity(dimsim_batch, "DIMSIM")
    if q_bundle != d_bundle:
        raise ValueError("Anchor-Q and DIMSIM bundle hashes differ")
    if q_models != d_models:
        raise ValueError("Anchor-Q and DIMSIM canonical model inventories differ")
    try:
        reference = canonical_model_id(reference_model)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference model is invalid") from exc
    if reference != reference_model:
        raise ValueError("reference model must be canonical")
    if run_ids is not None and (not isinstance(run_ids, dict) or set(run_ids) != {"contextual-anchor-q", "contextual-dimsim"}):
        raise ValueError("run_ids must map both analysis names to explicit safe IDs")

    pins = []
    for model in q_models:
        for seed in SEEDS:
            q, d = q_rows[(model, seed)], d_rows[(model, seed)]
            for field in ("run_id", "run_manifest_sha256", "checkpoint_sha256"):
                if q[field] != d[field]:
                    raise ValueError(f"Anchor-Q/DIMSIM {field} mismatch for {model} seed {seed}")
            qh, dh = q.get("experiment_hashes"), d.get("experiment_hashes")
            if qh is not None and dh is not None and qh != dh:
                raise ValueError(f"Anchor-Q/DIMSIM experiment_hashes mismatch for {model} seed {seed}")
            q_prov = q["score"].get("provenance")
            if not isinstance(q_prov, dict):
                raise ValueError(f"Anchor-Q score provenance missing for {model} seed {seed}")
            for field, expected in (("model_id", model), ("seed", seed),
                                    ("bundle_manifest_sha256", q_bundle),
                                    ("checkpoint_sha256", q["checkpoint_sha256"])):
                if q_prov.get(field) != expected:
                    raise ValueError(f"Anchor-Q outer/nested {field} mismatch for {model} seed {seed}")
            nested_hashes = q_prov.get("experiment_hashes")
            if (not isinstance(nested_hashes, dict) or set(nested_hashes) != _EXPERIMENT_HASH_NAMES
                    or any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in nested_hashes.values())):
                raise ValueError(f"Anchor-Q nested experiment_hashes inventory is invalid for {model} seed {seed}")
            hashes = qh if qh is not None else dh
            if hashes is not None and nested_hashes != hashes:
                raise ValueError(f"Anchor-Q outer/nested experiment_hashes mismatch for {model} seed {seed}")
            hashes = hashes if hashes is not None else nested_hashes
            pin = {"model_id": model, "seed": seed, "run_id": q["run_id"],
                   "run_manifest_sha256": q["run_manifest_sha256"], "checkpoint_sha256": q["checkpoint_sha256"],
                   "bundle_manifest_sha256": q_bundle,
                   "experiment_hashes": dict(sorted(hashes.items()))}
            pins.append(pin)
    for model in q_models:
        model_pins = [pin for pin in pins if pin["model_id"] == model]
        if any(pin["experiment_hashes"] != model_pins[0]["experiment_hashes"] for pin in model_pins[1:]):
            raise ValueError(f"resolved experiment_hashes differ across seeds for {model}")

    q_report = build_anchor_q_report(q_batch, reference_model=reference)
    d_report = build_dimsim_report(dimsim_batch, reference_model=reference)
    for name, report in (("Anchor-Q report", q_report), ("DIMSIM report", d_report)):
        _json_finite(report, name)
    _json_finite(q_batch, "Anchor-Q raw batch")
    _json_finite(dimsim_batch, "DIMSIM raw batch")
    figures = []
    for name, analysis, sources in _FIGURES:
        report = q_report if analysis == "contextual-anchor-q" else d_report
        data = {semantic_name: _semantic(report, key) for semantic_name, key in sources.items()}
        geometry = name == "DIMSIM geometry"
        figures.append({"name": name, "analysis": analysis, "data_keys": dict(sources),
                        "data": data, "axis_right_counts": geometry,
                        "ticks_labels_every_panel_both_columns": geometry,
                        "export_format": "unresolved",
                        "config_metadata": {"source_paths": dict(sources), "analysis": analysis}})

    artifact_plan = []
    for analysis in ("contextual-anchor-q", "contextual-dimsim"):
        report_name = "anchor_q_report.json" if analysis.endswith("anchor-q") else "dimsim_report.json"
        batch_name = "anchor_q_batch.json" if analysis.endswith("anchor-q") else "dimsim_batch.json"
        run_id = safe_run_id(run_ids[analysis]) if run_ids is not None else None
        if run_id is None:
            filenames, directories = None, None
        else:
            base = f"outputs/{analysis}/{run_id}"
            filenames = {"config": f"{base}/config.json", "manifest": f"{base}/manifest.json",
                         "input_batch": f"{base}/data/{batch_name}", "report": f"{base}/data/{report_name}",
                         "tables": f"{base}/tables/tables.json", "plot_data": f"{base}/figures/plot_data.json"}
            directories = [f"{base}/data", f"{base}/tables", f"{base}/figures", f"{base}/diagnostics"]
        artifact_plan.append({"analysis": analysis, "run_id": run_id,
            "path_template": f"outputs/{analysis}/{{run_id}}", "directories": directories, "filenames": filenames})

    result = {"contract_version": "1.0", "reference_model": reference,
        "scientific_input_pins": {"bundle_manifest_sha256": q_bundle, "model_ids": list(q_models),
                                  "seeds": list(SEEDS), "model_seed_provenance": pins},
        "validation_limits": {"dimsim_nested_provenance": "DIMSIM score schema carries model, seed and bundle identity but not checkpoint or experiment hashes; those pins are checked across the outer batches and retained explicitly."},
        "raw_batches": {"contextual-anchor-q": q_batch, "contextual-dimsim": dimsim_batch},
        "reports": {"contextual-anchor-q": q_report, "contextual-dimsim": d_report},
        "figures": figures, "artifact_plan": artifact_plan,
        "readiness": {"numeric_contract_valid": True, "five_seed_complete": True,
            "output_writer": "pending", "notebook_wiring": "pending", "extraction_wiring": "pending",
            "random_baseline_integration": "pending", "model_runtime": "pending", "figure_exports": "pending",
            "end_to_end_ready": False}}
    _json_finite(result, "evaluation report contract")
    return result
