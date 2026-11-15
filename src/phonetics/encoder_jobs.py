"""Config and input selection for encoder geometry scripts and notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import sha256_file
from .evaluation.encoder_extraction import EXTRACTION_TYPE, extract_hidden_states, load_experiment_manifest
from .objectives import canonical_model_id
from .execution import safe_id


def resolve_encoder_config(spec, resolve):
    required = {"id", "training_runs", "representations_root", "device", "batch_size", "tolerance"}
    if not isinstance(spec, dict) or not required <= set(spec) <= required | {"diagnostics", "contrasts", "allow_extraction", "execution"}:
        raise ValueError("encoder_geometry requires id, training_runs, representations_root, device, batch_size and tolerance")
    if type(spec.get("diagnostics", False)) is not bool:
        raise ValueError("encoder diagnostics must be a boolean")
    if type(spec.get("allow_extraction", False)) is not bool:
        raise ValueError("encoder allow_extraction must be a boolean")
    if spec.get("execution", "local") not in ("local", "notebook"):
        raise ValueError("encoder_geometry.execution must be local or notebook")
    safe_id(spec["id"])
    runs = spec["training_runs"]
    if (not isinstance(runs, dict) or not runs
            or any(canonical_model_id(model) != model for model in runs)):
        raise ValueError("training_runs requires canonical model IDs")
    if "contrasts" in spec:
        contrasts = spec["contrasts"]
        if (not isinstance(contrasts, list)
                or any(not isinstance(pair, list) or len(pair) != 2
                       or any(not isinstance(model, str) or model not in runs for model in pair)
                       or pair[0] == pair[1] for pair in contrasts)
                or len({tuple(pair) for pair in contrasts}) != len(contrasts)):
            raise ValueError("contrasts must contain unique pairs of distinct configured models")
    if not isinstance(spec["device"], str) or not spec["device"].strip():
        raise ValueError("encoder extraction device must be explicit")
    if type(spec["batch_size"]) is not int or spec["batch_size"] < 1:
        raise ValueError("encoder batch_size must be positive")
    import math
    if type(spec["tolerance"]) not in (int, float) or not math.isfinite(spec["tolerance"]) or spec["tolerance"] < 0:
        raise ValueError("encoder tolerance must be finite and nonnegative")
    return {**spec, "representations_root": resolve(spec["representations_root"]),
            "training_runs": {key: resolve(value) for key, value in spec["training_runs"].items()}}


def encoder_inputs(config, state=None):
    if "encoder_geometry" not in config or "evaluation" not in config:
        raise ValueError("configure evaluation and encoder_geometry first")
    from .execution import completed_manifest, evaluation_inputs, load_state

    managed = config["evaluation"]["experiment_manifest_sha256"] is None
    if managed and state is None:
        state = load_state(config)
    spec, evaluation = config["encoder_geometry"], evaluation_inputs(config, state)
    refs = {
        "evaluation": str(Path(evaluation["output_root"]) / evaluation["id"]),
        "experiment": evaluation["experiment_root"], **spec["training_runs"],
    }
    experiment = load_experiment_manifest(refs["experiment"], evaluation["experiment_manifest_sha256"])
    validate_run_inventory(refs, experiment)
    if managed:
        pins = {"experiment": evaluation["experiment_manifest_sha256"],
                "evaluation": completed_manifest(state, "prepare-evaluation", refs["evaluation"])}
        pins.update({model: completed_manifest(state, "train:" + model, path)
                     for model, path in spec["training_runs"].items()})
    else:
        pins = {key: sha256_file(Path(path) / "manifest.json") for key, path in refs.items()}
    if pins["experiment"] != evaluation["experiment_manifest_sha256"]:
        raise ValueError("configured experiment manifest changed")
    return refs, pins


def validate_run_inventory(refs, experiment):
    models = experiment["identity"]["config"]["model_ids"]
    required = {"experiment", "evaluation", *models}
    if set(refs) != required:
        raise ValueError("select exactly one training run for each experiment model")
    return models


def locate_manifests(refs, pins, mount):
    matches = {pin: set() for pin in pins.values()}
    paths = {Path(path).resolve() / "manifest.json" for path in refs.values()}
    if Path(mount).is_dir():
        paths.update(path.resolve() for path in Path(mount).rglob("manifest.json"))
    for path in sorted(paths):
        if path.is_file():
            pin = sha256_file(path)
            if pin in matches:
                matches[pin].add(path.parent)
    roots = {}
    for key, pin in pins.items():
        if len(matches[pin]) != 1:
            raise ValueError(f"attach exactly one {key} bundle with manifest SHA256 {pin}; found {len(matches[pin])}")
        roots[key] = next(iter(matches[pin]))
    return roots


def locate_exports(spec, pins, models, mount=None):
    preferred = Path(spec["representations_root"])
    manifests = {preferred / model / "manifest.json" for model in models}
    if mount is not None and Path(mount).is_dir():
        manifests.update(Path(mount).rglob("manifest.json"))
    result = {model: set() for model in models}
    for path in sorted(manifests):
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("bundle_type") != EXTRACTION_TYPE:
            continue
        identity = manifest.get("identity", {})
        model = identity.get("model")
        if (model in result and identity.get("evaluation_manifest_sha256") == pins["evaluation"]
                and identity.get("experiment_manifest_sha256") == pins["experiment"]
                and identity.get("training_manifest_sha256") == pins[model]):
            result[model].add(path.parent.resolve())
    if any(len(paths) > 1 for paths in result.values()):
        counts = {model: len(paths) for model, paths in result.items()}
        raise ValueError(f"multiple compatible hidden-state exports: {counts}")
    return {model: paths.pop() for model, paths in result.items() if paths}


def extract_configured(config, model, *, allow_execution):
    if not allow_execution:
        raise PermissionError("use --allow-execution after reviewing extraction inputs and device")
    from .bundles import content_hash
    from .execution import sdk_hashes

    refs, pins = encoder_inputs(config)
    if model not in config["encoder_geometry"]["training_runs"]:
        raise ValueError("unknown encoder model")
    spec = config["encoder_geometry"]
    return extract_hidden_states(
        evaluation_root=refs["evaluation"], evaluation_pin=pins["evaluation"],
        experiment_root=refs["experiment"], experiment_pin=pins["experiment"],
        training_root=refs[model], training_pin=pins[model], model=model,
        output=Path(spec["representations_root"]) / model, device=spec["device"],
        batch_size=spec["batch_size"], source_sha256=content_hash(sdk_hashes()), allow_execution=True,
    )


def import_hidden_states(config, model, result):
    import shutil
    from .bundles import staging_directory
    from .evaluation.encoder_extraction import expected_checkpoint_seeds, expected_unknown_positions, load_export, load_positions

    refs, pins = encoder_inputs(config)
    if model not in config["encoder_geometry"]["training_runs"]:
        raise ValueError("unknown encoder model")
    evaluation, rows = load_positions(refs["evaluation"], pins["evaluation"])
    if evaluation["identity"]["experiment_manifest_sha256"] != pins["experiment"]:
        raise ValueError("evaluation and experiment pins differ")
    experiment = load_experiment_manifest(refs["experiment"], pins["experiment"])
    expected_seeds = expected_checkpoint_seeds(refs[model], pins[model], model, pins["experiment"], experiment)
    unknown, hashes = expected_unknown_positions(
        refs["experiment"], pins["experiment"], rows, evaluation["identity"]["test_ids_sha256"])
    positions = [row["occurrence_id"] for row in rows]
    target = (Path(config["encoder_geometry"]["representations_root"]) / model).resolve()
    result = Path(result).resolve()
    if target.is_relative_to(result) or result.is_relative_to(target):
        raise ValueError("result and hidden-state output must be separate")
    def validate(root):
        for _ in load_export(root, model=model, evaluation_pin=pins["evaluation"],
                             experiment_pin=pins["experiment"], training_pin=pins[model], occurrence_ids=positions,
                             expected_seeds=expected_seeds, expected_unknown=unknown, expected_input_hashes=hashes):
            pass
    validate(result)
    with staging_directory(target) as stage:
        shutil.copytree(result, stage, dirs_exist_ok=True)
        validate(stage)
    return target


def run_configured_report(config, *, allow_execution=False, state=None):
    from .bundles import content_hash
    from .execution import sdk_hashes

    refs, pins = encoder_inputs(config, state)
    spec = config["encoder_geometry"]
    return run_encoder_workflow(refs=refs, pins=pins, spec=spec, reports_root=config["paths"]["reports"],
                                source_sha256=content_hash(sdk_hashes()), allow_execution=allow_execution)


def run_encoder_workflow(*, refs, pins, spec, reports_root, source_sha256, allow_execution=False, mount=None):
    from .evaluation.encoder_extraction import expected_checkpoint_seeds, expected_unknown_positions, load_export, load_positions
    from .evaluation.geometry_bundle import PAIRING_POLICY
    from .research.encoder_geometry import run_encoder_geometry
    from .visualization.core import require_arial

    roots = (locate_manifests(refs, pins, mount) if mount is not None
             else {key: Path(refs[key]) for key in pins})
    evaluation, occurrences = load_positions(roots["evaluation"], pins["evaluation"])
    if evaluation["identity"]["pairing_policy"] != PAIRING_POLICY:
        raise ValueError("encoder geometry requires the complete nonzero-initial evaluation bundle")
    if evaluation["identity"]["experiment_manifest_sha256"] != pins["experiment"]:
        raise ValueError("evaluation and experiment pins differ")
    experiment = load_experiment_manifest(roots["experiment"], pins["experiment"])
    models = validate_run_inventory(refs, experiment)
    validate_run_inventory(pins, experiment)
    exports = locate_exports(spec, pins, models, mount)
    missing = [model for model in models if model not in exports]
    if missing and not allow_execution:
        raise PermissionError(f"missing compatible exports for {', '.join(missing)}; authorize extraction first")
    require_arial()
    checkpoints = {model: expected_checkpoint_seeds(roots[model], pins[model], model, pins["experiment"], experiment)
                   for model in models}
    if missing and exports:
        positions = [row["occurrence_id"] for row in occurrences]
        unknown, hashes = expected_unknown_positions(
            roots["experiment"], pins["experiment"], occurrences, evaluation["identity"]["test_ids_sha256"])
        for model, root in exports.items():
            for _ in load_export(root, model=model, evaluation_pin=pins["evaluation"],
                                 experiment_pin=pins["experiment"], training_pin=pins[model], occurrence_ids=positions,
                                 expected_seeds=checkpoints[model], expected_unknown=unknown, expected_input_hashes=hashes):
                pass
    for model in missing:
        exports[model] = extract_hidden_states(
            evaluation_root=roots["evaluation"], evaluation_pin=pins["evaluation"],
            experiment_root=roots["experiment"], experiment_pin=pins["experiment"],
            training_root=roots[model], training_pin=pins[model], model=model,
            output=Path(spec["representations_root"]) / model, device=spec["device"],
            batch_size=spec["batch_size"], source_sha256=source_sha256, allow_execution=allow_execution,
        )
    return run_encoder_geometry(evaluation_root=roots["evaluation"], experiment_root=roots["experiment"],
                                pins=pins, exports=exports, checkpoints=checkpoints,
                                output=Path(reports_root) / spec["id"],
                                tolerance=spec["tolerance"], source_sha256=source_sha256,
                                diagnostics=spec.get("diagnostics", False), contrasts=spec.get("contrasts"))
