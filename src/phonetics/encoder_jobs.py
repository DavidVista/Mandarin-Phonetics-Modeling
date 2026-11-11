"""Config and input selection for encoder geometry scripts and notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import sha256_file
from .evaluation.encoder_extraction import EXTRACTION_TYPE, extract_hidden_states
from .evaluation.training_summary import TRAINING_MODELS
from .execution import safe_id


def resolve_encoder_config(spec, resolve):
    required = {"id", "training_runs", "representations_root", "device", "batch_size", "tolerance"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("encoder_geometry requires id, training_runs, representations_root, device, batch_size and tolerance")
    safe_id(spec["id"])
    if not isinstance(spec["training_runs"], dict) or set(spec["training_runs"]) != set(TRAINING_MODELS):
        raise ValueError("select exactly one training run for each encoder model")
    if not isinstance(spec["device"], str) or not spec["device"].strip():
        raise ValueError("encoder extraction device must be explicit")
    if type(spec["batch_size"]) is not int or spec["batch_size"] < 1:
        raise ValueError("encoder batch_size must be positive")
    import math
    if type(spec["tolerance"]) not in (int, float) or not math.isfinite(spec["tolerance"]) or spec["tolerance"] < 0:
        raise ValueError("encoder tolerance must be finite and nonnegative")
    return {**spec, "representations_root": resolve(spec["representations_root"]),
            "training_runs": {key: resolve(value) for key, value in spec["training_runs"].items()}}


def encoder_inputs(config):
    if "encoder_geometry" not in config or "evaluation" not in config:
        raise ValueError("configure evaluation and encoder_geometry first")
    spec, evaluation = config["encoder_geometry"], config["evaluation"]
    refs = {
        "evaluation": str(Path(evaluation["output_root"]) / evaluation["id"]),
        "experiment": evaluation["experiment_root"], **spec["training_runs"],
    }
    pins = {key: sha256_file(Path(path) / "manifest.json") for key, path in refs.items()}
    if pins["experiment"] != evaluation["experiment_manifest_sha256"]:
        raise ValueError("configured experiment manifest changed")
    return refs, pins


def locate_manifest(pin, preferred, mount="/kaggle/input"):
    paths = {Path(preferred).resolve()}
    if Path(mount).is_dir():
        paths.update(path.parent.resolve() for path in Path(mount).rglob("manifest.json"))
    matches = sorted(path for path in paths if (path / "manifest.json").is_file()
                     and sha256_file(path / "manifest.json") == pin)
    if len(matches) != 1:
        raise ValueError(f"attach exactly one bundle with manifest SHA256 {pin}; found {len(matches)}")
    return matches[0]


def locate_exports(spec, pins, mount="/kaggle/input"):
    preferred = Path(spec["representations_root"])
    manifests = {preferred / model / "manifest.json" for model in TRAINING_MODELS}
    if Path(mount).is_dir():
        manifests.update(Path(mount).rglob("manifest.json"))
    result = {model: set() for model in TRAINING_MODELS}
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
    if any(len(paths) != 1 for paths in result.values()):
        counts = {model: len(paths) for model, paths in result.items()}
        raise ValueError(f"attach one compatible hidden-state export per model: {counts}")
    return {model: paths.pop() for model, paths in result.items()}


def extract_configured(config, model, *, allow_execution):
    if not allow_execution:
        raise PermissionError("use --allow-execution after reviewing extraction inputs and device")
    if model not in TRAINING_MODELS:
        raise ValueError("unknown encoder model")
    from .bundles import content_hash
    from .execution import sdk_hashes

    refs, pins = encoder_inputs(config)
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
    from .evaluation.encoder_extraction import inspect_checkpoints, load_export, load_positions

    if model not in TRAINING_MODELS:
        raise ValueError("unknown encoder model")
    refs, pins = encoder_inputs(config)
    _, rows = load_positions(refs["evaluation"], pins["evaluation"])
    training, selected = inspect_checkpoints(refs[model], pins[model], model, pins["experiment"])
    expected_seeds = {seed: {"checkpoint_sha256": training["artifacts"][f"seed-{seed}/best.pt"],
                             "seed_input_hashes": item["seed_input_hashes"], "best_epoch": item["best_epoch"]}
                      for seed, item in selected.items()}
    positions = [row["occurrence_id"] for row in rows]
    target = (Path(config["encoder_geometry"]["representations_root"]) / model).resolve()
    result = Path(result).resolve()
    if target.is_relative_to(result) or result.is_relative_to(target):
        raise ValueError("result and hidden-state output must be separate")
    def validate(root):
        for _ in load_export(root, model=model, evaluation_pin=pins["evaluation"],
                             experiment_pin=pins["experiment"], training_pin=pins[model], occurrence_ids=positions,
                             expected_seeds=expected_seeds):
            pass
    validate(result)
    with staging_directory(target) as stage:
        shutil.copytree(result, stage, dirs_exist_ok=True)
        validate(stage)
    return target


def run_configured_report(config):
    from .bundles import content_hash
    from .execution import sdk_hashes
    from .research.encoder_geometry import run_encoder_geometry

    refs, pins = encoder_inputs(config)
    spec = config["encoder_geometry"]
    exports = locate_exports(spec, pins)
    return run_encoder_geometry(evaluation_root=refs["evaluation"], pins=pins, exports=exports,
                                output=Path(config["paths"]["reports"]) / spec["id"],
                                tolerance=spec["tolerance"], source_sha256=content_hash(sdk_hashes()))
