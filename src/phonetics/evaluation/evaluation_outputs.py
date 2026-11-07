"""Immutable, hash-verified persistence for contextual evaluation outputs."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from ..research.artifacts import file_sha256, safe_analysis, safe_run_id, ensure_output_outside_inputs


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json(value))


def _tree_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _finite_json(path: Path):
    def bad(token): raise ValueError(f"non-finite JSON constant: {token}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)
    _json(value)
    return value


def _pins(inputs, scientific_pins):
    bundle_pin = scientific_pins.get("bundle_manifest_sha256")
    if bundle_pin and bundle_pin not in {x["sha256"] for x in inputs}:
        raise ValueError("scientific bundle manifest pin does not match an input identity")
    exp_pin = scientific_pins.get("experiment_manifest_sha256")
    if exp_pin and exp_pin not in {x["sha256"] for x in inputs}:
        raise ValueError("scientific experiment manifest pin does not match an input identity")


def _verify_inventory(root: Path, analysis: str, run_id: str, expected: set[str], *,
                      profile: str = "numerical_outputs", figure_exports: str = "pending"):
    safe_analysis(analysis); safe_run_id(run_id)
    manifest = _finite_json(root / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("analysis") != analysis or manifest.get("run_id") != run_id
            or manifest.get("scope") != profile or manifest.get("figure_exports") != figure_exports
            or set(manifest.get("required_artifacts", [])) != expected):
        raise ValueError("output manifest identity/profile mismatch")
    if (not isinstance(manifest.get("python"), str) or not manifest["python"]
            or not isinstance(manifest.get("dependencies"), dict) or "numpy" not in manifest["dependencies"]
            or not isinstance(manifest.get("inputs"), list) or not isinstance(manifest.get("source_hashes"), dict)
            or not manifest["source_hashes"]):
        raise ValueError("output provenance metadata is incomplete")
    for row in manifest["inputs"]:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not isinstance(row.get("identity_path"), str) or not _is_sha(row.get("sha256"))):
            raise ValueError("output input identity metadata is invalid")
    if any(not _is_sha(x) for x in manifest["source_hashes"].values()):
        raise ValueError("output source hash metadata is invalid")
    actual, inventory = _tree_files(root) - {"manifest.json"}, manifest.get("artifacts")
    declared = {x.get("path") for x in inventory if isinstance(x, dict)} if isinstance(inventory, list) else set()
    if actual != expected or declared != expected or len(inventory or []) != len(expected):
        raise ValueError("output artifact inventory mismatch")
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            raise ValueError("malformed artifact inventory row")
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"artifact hash or size mismatch: {row.get('path')}")
    return manifest


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@contextmanager
def _evaluation_run(project_root: str | Path, analysis: str, run_id: str, *,
                    input_roots: list[str | Path], config: dict, source_hashes: dict,
                    expected_files: set[str], scientific_pins: dict, create_data: bool = True,
                    output_path: str | Path | None = None, profile: str = "numerical_outputs",
                    figure_exports_on_complete: str = "pending"):
    """Write one run, gating every file except its top-level manifest."""
    analysis, run_id = safe_analysis(analysis), safe_run_id(run_id)
    if figure_exports_on_complete not in {"pending", "complete"}:
        raise ValueError("unsupported figure export completion status")
    root = Path(output_path).resolve() if output_path is not None else Path(project_root).resolve() / "outputs" / analysis / run_id
    ensure_output_outside_inputs(root, input_roots)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {root}")
    if not isinstance(config, dict) or not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("explicit config and source hash map are required")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in source_hashes.values()):
        raise ValueError("source hashes must be lowercase SHA-256 digests")
    # Validate all metadata before creating the destination.
    _json({"config": config, "source_hashes": source_hashes, "scientific_pins": scientific_pins})
    inputs = []
    for raw in input_roots:
        p = Path(raw).resolve(strict=True)
        identity = p if p.is_file() else p / "manifest.json"
        if not identity.is_file():
            raise FileNotFoundError(f"input manifest identity missing: {identity}")
        inputs.append({"path": str(p), "identity_path": str(identity), "sha256": file_sha256(identity)})
    _pins(inputs, scientific_pins)
    dependencies = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib"):
        try: dependencies[package] = version(package)
        except PackageNotFoundError: dependencies[package] = None
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "analysis": analysis, "run_id": run_id,
                "status": "running", "scope": profile, "figure_exports": "pending",
                "required_artifacts": sorted(expected_files), "artifacts": [],
                "scientific_input_pins": scientific_pins, "source_hashes": source_hashes,
                "inputs": inputs, "python": platform.python_version(), "dependencies": dependencies}
    _write(root / "manifest.json", manifest)
    try:
        names = (("data",) if create_data else ()) + ("tables", "figures", "diagnostics")
        for name in names:
            (root / name).mkdir()
        _write(root / "config.json", config)
        yield root
        actual = _tree_files(root) - {"manifest.json"}
        if actual != expected_files:
            raise RuntimeError(f"artifact inventory mismatch; missing={sorted(expected_files-actual)}, unexpected={sorted(actual-expected_files)}")
        artifacts = []
        for relative in sorted(actual):
            path = root / relative
            if path.stat().st_size <= 0:
                raise RuntimeError(f"required artifact is empty: {relative}")
            artifacts.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
        manifest.update(status="complete", artifacts=artifacts, figure_exports=figure_exports_on_complete)
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        fd, name = tempfile.mkstemp(prefix="manifest-", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_json(manifest))
            os.replace(name, root / "manifest.json")
        finally:
            if os.path.exists(name):
                os.unlink(name)


def write_evaluation_outputs(project_root: str | Path, contract: dict, *, run_ids: dict[str, str],
                             input_roots: list[str | Path], source_hashes: dict[str, str],
                             resolved_config: dict) -> dict[str, Path]:
    """Persist both report-contract analyses to their explicit immutable run IDs."""
    from .report_contract import build_evaluation_report
    if not isinstance(contract, dict) or not isinstance(run_ids, dict) or set(run_ids) != {"contextual-anchor-q", "contextual-dimsim"}:
        raise ValueError("contract and explicit IDs for both analyses are required")
    rebuilt = build_evaluation_report(contract["raw_batches"]["contextual-anchor-q"],
                                      contract["raw_batches"]["contextual-dimsim"],
                                      contract["reference_model"], run_ids)
    # The contract is a derived view; refuse stale/tampered reports or pins.
    for key in ("scientific_input_pins", "reports", "figures"):
        if contract.get(key) != rebuilt.get(key):
            raise ValueError(f"evaluation contract {key} is stale or inconsistent")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("explicit source code hash map is required")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in source_hashes.values()):
        raise ValueError("source hashes must be lowercase SHA-256 digests")
    if not isinstance(resolved_config, dict):
        raise ValueError("resolved config is required")
    # Validate every target before either analysis creates a directory.
    project = Path(project_root).resolve()
    target_roots = []
    for name in ("contextual-anchor-q", "contextual-dimsim"):
        target = project / "outputs" / safe_analysis(name) / safe_run_id(run_ids[name])
        ensure_output_outside_inputs(target, input_roots)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
        if any(target == prior or target in prior.parents or prior in target.parents for prior in target_roots):
            raise ValueError("evaluation output targets overlap")
        target_roots.append(target)
    basepins = contract["scientific_input_pins"]
    results = {}
    for analysis in ("contextual-anchor-q", "contextual-dimsim"):
        run_id = safe_run_id(run_ids[analysis])
        q = analysis.endswith("anchor-q")
        batch_name = "anchor_q_batch.json" if q else "dimsim_batch.json"
        report_name = "anchor_q_report.json" if q else "dimsim_report.json"
        report = rebuilt["reports"][analysis]
        batch = rebuilt["raw_batches"][analysis]
        tables, plot = report["tables"], report["plot_data"]
        expected = {"config.json", f"data/{batch_name}", f"data/{report_name}", "tables/tables.json",
                    "figures/plot_data.json", "diagnostics/summary.json"}
        pins = {**basepins, "analysis": analysis, "run_id": run_id}
        with _evaluation_run(project_root, analysis, run_id, input_roots=input_roots, config=resolved_config,
                             source_hashes=source_hashes, expected_files=expected, scientific_pins=pins) as root:
            _write(root / f"data/{batch_name}", batch)
            _write(root / f"data/{report_name}", report)
            _write(root / "tables/tables.json", tables)
            _write(root / "figures/plot_data.json", plot)
            _write(root / "diagnostics/summary.json", {"scope": "numerical_outputs", "figure_exports": "pending",
                    "model_count": len(basepins["model_ids"]), "seed_ids": basepins["seeds"],
                    "table_row_counts": {name: len(rows) for name, rows in tables.items()},
                    "plot_data_keys": sorted(plot)})
        results[analysis] = root
    return results


def load_evaluation_summary(root: str | Path) -> dict:
    """Verify every artifact hash but load only the small report views."""
    root = Path(root).resolve(strict=True)
    analysis = root.parent.name
    if analysis not in {'contextual-anchor-q', 'contextual-dimsim'}:
        raise ValueError('unknown evaluation output analysis')
    name = 'anchor_q' if analysis == 'contextual-anchor-q' else 'dimsim'
    expected = {'config.json', f'data/{name}_batch.json', f'data/{name}_report.json',
                'tables/tables.json', 'figures/plot_data.json', 'diagnostics/summary.json'}
    manifest = _verify_inventory(root, analysis, root.name, expected)
    _pins(manifest.get('inputs', []), manifest.get('scientific_input_pins', {}))
    report = _finite_json(root / f'data/{name}_report.json')
    tables = _finite_json(root / 'tables/tables.json')
    plot = _finite_json(root / 'figures/plot_data.json')
    if tables != report.get('tables') or plot != report.get('plot_data'):
        raise ValueError('stored report views disagree')
    return {'root': root, 'manifest': manifest, 'report': report, 'tables': tables,
            'plot_data': plot, 'config': _finite_json(root / 'config.json')}


def load_evaluation_outputs(root: str | Path) -> dict:
    """Load a complete numerical run after checking exact inventory and hashes."""
    root = Path(root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    analysis = root.parent.name
    if analysis not in {"contextual-anchor-q", "contextual-dimsim"}: raise ValueError("unknown evaluation output analysis")
    q = analysis.endswith("anchor-q")
    bn, rn = ("anchor_q_batch.json", "anchor_q_report.json") if q else ("dimsim_batch.json", "dimsim_report.json")
    expected = {"config.json", f"data/{bn}", f"data/{rn}", "tables/tables.json", "figures/plot_data.json", "diagnostics/summary.json"}
    manifest = _verify_inventory(root, analysis, root.name, expected)
    _pins(manifest.get("inputs", []), manifest.get("scientific_input_pins", {}))
    if set(manifest["scientific_input_pins"]) != {"bundle_manifest_sha256", "model_ids", "seeds", "model_seed_provenance", "analysis", "run_id"}:
        raise ValueError("numerical output pins are malformed")
    batch, report = _finite_json(root / f"data/{bn}"), _finite_json(root / f"data/{rn}")
    tables, plot = _finite_json(root / "tables/tables.json"), _finite_json(root / "figures/plot_data.json")
    config = _finite_json(root / "config.json"); _finite_json(root / "diagnostics/summary.json")
    if tables != report.get("tables") or plot != report.get("plot_data"):
        raise ValueError("stored tables or plot data differ from stored report")
    if batch.get("bundle_manifest_sha256") != manifest["scientific_input_pins"]["bundle_manifest_sha256"]:
        raise ValueError("stored batch bundle pin mismatch")
    results = batch.get("results")
    pins = manifest["scientific_input_pins"]
    if not isinstance(results, list) or sorted({r.get("model_id") for r in results if isinstance(r, dict)}) != pins["model_ids"]:
        raise ValueError("stored batch model inventory differs from manifest pins")
    if any(not isinstance(r, dict) or type(r.get("seed")) is not int or r.get("seed") not in pins["seeds"] for r in results):
        raise ValueError("stored batch seed inventory is invalid")
    identity_pins = {(p.get("model_id"), p.get("seed")): p for p in pins["model_seed_provenance"] if isinstance(p, dict)}
    expected_pairs = {(model, seed) for model in pins["model_ids"] for seed in pins["seeds"]}
    if len(identity_pins) != len(pins["model_seed_provenance"]) or set(identity_pins) != expected_pairs:
        raise ValueError("manifest model/seed provenance inventory is invalid")
    for row in results:
        pin = identity_pins[(row["model_id"], row["seed"])]
        if any(row.get(k) != pin.get(k) for k in ("run_id", "run_manifest_sha256", "checkpoint_sha256")):
            raise ValueError("stored batch provenance differs from manifest pins")
        if row.get("experiment_hashes") is not None and row["experiment_hashes"] != pin.get("experiment_hashes"):
            raise ValueError("stored batch experiment hashes differ from manifest pins")
    if config.get("reference_model") is not None and report.get("reference_model") is not None and report.get("reference_model") != config["reference_model"]:
        raise ValueError("stored report reference metadata differs from config")
    return {"root": root, "manifest": manifest, "config": config,
            "batch": batch, "report": report, "tables": tables, "plot_data": plot}


def write_random_control_outputs(project_root: str | Path, run_id: str, control: dict, *,
                                input_roots: list[str | Path], source_hashes: dict[str, str],
                                resolved_config: dict) -> Path:
    """Persist an already evaluated random control without loading learned representations."""
    import io
    import numpy as np
    from .random_control import RANDOM_SEEDS
    if not isinstance(control, dict) or not isinstance(control.get("report"), dict):
        raise ValueError("evaluated random-control result is required")
    ids, matrices, report = control.get("occurrence_ids"), control.get("matrices"), control["report"]
    if (not isinstance(ids, list) or not ids or ids != sorted(ids) or len(ids) != len(set(ids))
            or any(not isinstance(x, str) or not x for x in ids)
            or not isinstance(matrices, dict) or any(type(s) is not int for s in matrices)
            or set(matrices) != set(RANDOM_SEEDS)):
        raise ValueError("random-control IDs and five matrices are required")
    dimension = None; normalized_matrices = {}
    for seed in RANDOM_SEEDS:
        matrix = np.asarray(matrices[seed])
        if matrix.dtype != np.dtype("float32") or matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
            raise ValueError(f"random matrix {seed} violates finite float32 shape contract")
        if dimension is None: dimension = matrix.shape[1]
        if matrix.shape[1] != dimension or np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) == 0):
            raise ValueError("random matrix dimensions or norms are invalid")
        normalized_matrices[seed] = matrix
    if (report.get("seed_kind") != "random" or report.get("model_id") != "random-control"
            or report.get("random_seeds") != list(RANDOM_SEEDS)):
        raise ValueError("control report identity or seed-kind is invalid")
    from .random_control import generate_random_control, summarize_random_control
    if not _is_sha(report.get("bundle_manifest_sha256")) or not isinstance(report.get("per_seed"), list) or len(report["per_seed"]) != 5:
        raise ValueError("random-control report bundle pin and five seed records are required")
    regenerated = generate_random_control(ids, dimension=int(dimension))
    for seed in RANDOM_SEEDS:
        if normalized_matrices[seed].tobytes(order="C") != regenerated["matrices"][seed].tobytes(order="C"):
            raise ValueError(f"random matrix {seed} does not match its declared PCG64 realization")
        rows = report["per_seed"]
        if (not isinstance(rows[seed], dict) or rows[seed].get("random_seed") != seed
                or rows[seed].get("provenance", {}).get("dimension") != dimension
                or rows[seed].get("provenance", {}).get("bundle_manifest_sha256") != report.get("bundle_manifest_sha256")):
            raise ValueError("random report seed, dimension, or bundle metadata mismatch")
    try: summary_matches = _json(summarize_random_control(report["per_seed"], report["bundle_manifest_sha256"])) == _json(report)
    except (KeyError, TypeError, ValueError) as exc: raise ValueError("random-control per-seed report is invalid") from exc
    if not summary_matches: raise ValueError("random-control report is stale relative to its per-seed records")
    # Top-level output preflight and run directory creation.
    target = Path(project_root).resolve() / "outputs" / "contextual-random-control" / safe_run_id(run_id)
    ensure_output_outside_inputs(target, input_roots)
    if target.exists(): raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    if not isinstance(resolved_config, dict): raise ValueError("resolved config is required")
    expected = {"config.json", "data/occurrence_ids.json", "data/control_report.json", "tables/tables.json",
                "figures/plot_data.json", "diagnostics/summary.json", *(f"data/random_seed_{s}.npy" for s in RANDOM_SEEDS)}
    pins = {"bundle_manifest_sha256": report.get("bundle_manifest_sha256"), "seed_kind": "random",
            "random_seeds": list(RANDOM_SEEDS), "dimension": int(dimension)}
    with _evaluation_run(project_root, "contextual-random-control", run_id, input_roots=input_roots,
                         config=resolved_config, source_hashes=source_hashes,
                         expected_files=expected, scientific_pins=pins) as root:
        _write(root / "data/occurrence_ids.json", ids)
        _write(root / "data/control_report.json", report)
        for seed in RANDOM_SEEDS:
            buf = io.BytesIO(); np.save(buf, matrices[seed], allow_pickle=False)
            (root / f"data/random_seed_{seed}.npy").write_bytes(buf.getvalue())
        q, d = report.get("anchor_q", {}), report.get("dimsim", {})
        tables = {"anchor_q": q, "dimsim": d}
        plot = {"anchor_q": q, "dimsim": d}
        _write(root / "tables/tables.json", tables)
        _write(root / "figures/plot_data.json", plot)
        _write(root / "diagnostics/summary.json", {"scope": "numerical_outputs", "figure_exports": "pending",
                "occurrence_count": len(ids), "dimension": int(dimension), "random_seeds": list(RANDOM_SEEDS)})
    return target


def load_random_control_summary(root: str | Path) -> dict:
    """Read hash-verified random-control summaries without loading vector matrices."""
    root = Path(root).resolve(strict=True)
    expected = {"config.json", "data/occurrence_ids.json", "data/control_report.json",
                "tables/tables.json", "figures/plot_data.json", "diagnostics/summary.json",
                *(f"data/random_seed_{seed}.npy" for seed in range(5))}
    manifest = _verify_inventory(root, "contextual-random-control", root.name, expected)
    pins = manifest.get("scientific_input_pins", {})
    _pins(manifest.get("inputs", []), pins)
    if (set(pins) != {"bundle_manifest_sha256", "seed_kind", "random_seeds", "dimension"}
            or pins["seed_kind"] != "random" or pins["random_seeds"] != list(range(5))
            or type(pins["dimension"]) is not int or pins["dimension"] <= 0):
        raise ValueError("random-control scientific pins malformed")
    report = _finite_json(root / "data/control_report.json")
    if (report.get("schema_version") != 1 or report.get("model_id") != "random-control"
            or report.get("seed_kind") != "random" or report.get("random_seeds") != list(range(5))
            or report.get("bundle_manifest_sha256") != pins["bundle_manifest_sha256"]):
        raise ValueError("stored random-control identity mismatch")
    tables = _finite_json(root / "tables/tables.json")
    plot = _finite_json(root / "figures/plot_data.json")
    if tables != {"anchor_q": report.get("anchor_q"), "dimsim": report.get("dimsim")} or plot != tables:
        raise ValueError("random-control table or plot payload mismatch")
    return {"root": root, "manifest": manifest, "config": _finite_json(root / "config.json"),
            "report": {key: value for key, value in report.items() if key != "per_seed"},
            "tables": tables, "plot_data": plot}


def load_random_control_outputs(root: str | Path) -> dict:
    """Recover persisted Gaussian controls directly; this never invokes learned loaders."""
    import numpy as np
    root = Path(root).resolve(strict=True)
    expected = {"config.json", "data/occurrence_ids.json", "data/control_report.json", "tables/tables.json", "figures/plot_data.json", "diagnostics/summary.json", *(f"data/random_seed_{s}.npy" for s in range(5))}
    manifest = _verify_inventory(root, "contextual-random-control", root.name, expected)
    _pins(manifest.get("inputs", []), manifest.get("scientific_input_pins", {}))
    if set(manifest["scientific_input_pins"]) != {"bundle_manifest_sha256", "seed_kind", "random_seeds", "dimension"}:
        raise ValueError("random-control scientific pins malformed")
    ids = _finite_json(root / "data/occurrence_ids.json")
    matrices = {seed: np.load(root / f"data/random_seed_{seed}.npy", allow_pickle=False)
                for seed in range(5)}
    report = _finite_json(root / "data/control_report.json")
    if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids)
            or ids != sorted(ids) or len(ids) != len(set(ids))
            or report.get("model_id") != "random-control" or report.get("seed_kind") != "random"
            or report.get("bundle_manifest_sha256") != manifest["scientific_input_pins"]["bundle_manifest_sha256"]
            or report.get("random_seeds") != [0, 1, 2, 3, 4]):
        raise ValueError("stored random-control identity mismatch")
    dimension = manifest["scientific_input_pins"]["dimension"]
    from .random_control import generate_random_control, summarize_random_control
    expected_matrices = generate_random_control(ids, dimension=dimension)["matrices"]
    for seed, matrix in matrices.items():
        if (matrix.dtype != np.dtype("float32") or matrix.shape != (len(ids), dimension) or not np.isfinite(matrix).all()
                or np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) == 0)
                or matrix.tobytes(order="C") != expected_matrices[seed].tobytes(order="C")):
            raise ValueError(f"stored random matrix {seed} violates declared realization")
    if _json(summarize_random_control(report["per_seed"], report["bundle_manifest_sha256"])) != _json(report):
        raise ValueError("stored random-control report summary is stale")
    tables, plot = _finite_json(root / "tables/tables.json"), _finite_json(root / "figures/plot_data.json")
    if tables != {"anchor_q": report["anchor_q"], "dimsim": report["dimsim"]} or plot != tables:
        raise ValueError("random-control table or plot payload mismatch")
    _finite_json(root / "config.json"); _finite_json(root / "diagnostics/summary.json")
    return {"manifest": manifest, "config": _finite_json(root / "config.json"),
            "occurrence_ids": ids, "matrices": matrices, "report": report, "tables": tables, "plot_data": plot}


def run_contextual_extraction_outputs(plan: dict, *, extractor=None) -> dict:
    """Execute an explicit extraction plan and gate each raw export after validation."""
    from contextlib import contextmanager
    from .extraction_batch import execute_contextual_extraction, _revalidate_plan
    plan = _revalidate_plan(plan)
    inputs = [plan["bundle_path"], plan["experiment_path"], *plan["training_runs"].values()]
    source_hashes = {"evaluation/encoders.py": plan["source_sha256"]}
    contexts = {}
    for item in plan["inventory"]:
        run_id = safe_run_id(item["output_run_id"])
        root = Path(item["output_run_dir"])
        pins = {"bundle_manifest_sha256": plan["bundle_manifest_sha256"],
                "experiment_manifest_sha256": plan["experiment_manifest_sha256"],
                "experiment_hashes": plan["experiment_hashes"], "model_id": item["model_id"],
                "seed": item["seed"], "checkpoint_sha256": item["checkpoint_sha256"],
                "training_manifest_sha256": item["training_manifest_sha256"]}
        expected = {"config.json", "data/manifest.json", "data/occurrence_ids.json",
                    "data/occurrence_hidden.npy", "diagnostics/summary.json"}
        config = {"mode": "contextual_only", "device": plan["device"], "batch_size": plan["batch_size"],
                  "model_id": item["model_id"], "seed": item["seed"], "checkpoint_sha256": item["checkpoint_sha256"]}
        @contextmanager
        def lifecycle(item=item, root=root, run_id=run_id, pins=pins, expected=expected, config=config):
            with _evaluation_run(plan["output_root"], "contextual-representations", run_id,
                                 input_roots=inputs, config=config, source_hashes=source_hashes,
                                 expected_files=expected, scientific_pins=pins, create_data=False,
                                 output_path=root, profile="contextual_representation_export") as runroot:
                yield
                _write(runroot / "diagnostics/summary.json", {"model_id": item["model_id"], "seed": item["seed"],
                        "representation": "contextual_hidden_prehead", "validated_after_export": True,
                        "nested_manifest_sha256": file_sha256(root / "data/manifest.json")})
        contexts[item["output_run_id"]] = lifecycle
    metadata = execute_contextual_extraction(plan, extractor=extractor,
        context_factory=lambda item: contexts[item["output_run_id"]]())
    return metadata


def load_contextual_extraction_outputs(root: str | Path) -> dict:
    """Recover a validated raw contextual export including its nested manifest pin."""
    import numpy as np
    root = Path(root).resolve(strict=True)
    expected = {"config.json", "data/manifest.json", "data/occurrence_ids.json",
                "data/occurrence_hidden.npy", "diagnostics/summary.json"}
    manifest = _verify_inventory(root, "contextual-representations", root.name, expected,
                                 profile="contextual_representation_export")
    from .representations import load_contextual_representations
    nested_export = load_contextual_representations(root / "data")
    nested = nested_export["manifest"]
    diagnostic = _finite_json(root / "diagnostics/summary.json")
    pins = manifest.get("scientific_input_pins", {})
    provenance = nested.get("provenance", {})
    if (diagnostic.get("nested_manifest_sha256") != file_sha256(root / "data/manifest.json")
            or provenance.get("model_id") != pins.get("model_id") or provenance.get("seed") != pins.get("seed")
            or provenance.get("checkpoint_sha256") != pins.get("checkpoint_sha256")
            or provenance.get("bundle_manifest_sha256") != pins.get("bundle_manifest_sha256")
            or provenance.get("experiment_hashes") != pins.get("experiment_hashes")
            or provenance.get("source_sha256") != manifest.get("source_hashes", {}).get("evaluation/encoders.py")):
        raise ValueError("nested contextual manifest pin mismatch")
    _pins(manifest.get("inputs", []), pins)
    return {"manifest": manifest, "config": _finite_json(root / "config.json"),
            "occurrence_ids": json.loads((root / "data/occurrence_ids.json").read_text(encoding="utf-8")),
            "hidden": nested_export["hidden"],
            "representation_manifest": nested, "diagnostics": diagnostic}
