"""Pinned, read-only planning and execution for contextual extraction batches."""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from ..objectives import canonical_model_id
from .bundle import load_evaluation_bundle
from .seed_runs import discover_seed_runs

_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_EXP_FILES = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"}


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _exp_hashes(root: Path, manifest_hash: str) -> dict[str, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not _digest(artifacts.get("acoustic_targets.npy")):
        raise ValueError(f"experiment manifest lacks acoustic_targets.npy hash: {root}")
    result = {name: manifest_hash if name == "manifest.json" else artifacts.get(name)
              for name in _EXP_FILES}
    for name, digest in result.items():
        path = root / name
        if not _digest(digest) or not path.is_file() or _hash(path) != digest:
            raise ValueError(f"experiment source hash mismatch: {root / name}")
    return result


def resolve_encoder_source_hash(build_metadata: dict[str, Any]) -> str:
    """Resolve the encoder digest from an explicit verified build record."""
    hashes = build_metadata.get("source_sha256") if isinstance(build_metadata, dict) else None
    value = hashes.get("evaluation/encoders.py") if isinstance(hashes, dict) else None
    if not _digest(value):
        raise ValueError("build metadata lacks evaluation/encoders.py SHA-256")
    return value


def _safe_id(value: Any) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    return (isinstance(value, str) and _ID.fullmatch(value) is not None and ".." not in value
            and not value.endswith((".", " ")) and value.upper().split(".")[0] not in reserved)


def plan_contextual_extraction(*, bundle_path: str | Path, bundle_manifest_sha256: str,
                               experiment_path: str | Path, experiment_manifest_sha256: str,
                               training_runs: dict[str, str | Path], run_ids: dict[str, dict[int, str]],
                               output_root: str | Path, device: str = "cpu", batch_size: int = 32,
                               expected_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
                               source_sha256: str) -> dict[str, Any]:
    """Validate all 25+ selected runs and return a deterministic JSON-compatible plan."""
    if not _digest(bundle_manifest_sha256) or not _digest(experiment_manifest_sha256) or not _digest(source_sha256):
        raise ValueError("manifest and source pins must be lowercase SHA-256 digests")
    if tuple(expected_seeds) != (0, 1, 2, 3, 4) or any(type(x) is not int for x in expected_seeds):
        raise ValueError("expected_seeds must be exactly integer seeds 0 through 4")
    if not isinstance(device, str) or not device.strip() or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("device and positive integer batch_size are required")
    if not isinstance(training_runs, dict) or not training_runs or not isinstance(run_ids, dict):
        raise ValueError("training_runs and run_ids mappings are required")
    canonical = {}
    for key, path in training_runs.items():
        model = canonical_model_id(key)
        if key != model or model in canonical:
            raise ValueError("training_runs keys must be unique canonical model IDs")
        canonical[model] = path
    models = sorted(canonical)
    if set(run_ids) != set(models):
        raise ValueError("run_ids must contain exactly the canonical training models")
    bundle_root, exp_root = Path(bundle_path).resolve(strict=True), Path(experiment_path).resolve(strict=True)
    if _hash(bundle_root / "manifest.json") != bundle_manifest_sha256:
        raise ValueError("bundle manifest pin mismatch")
    bundle = load_evaluation_bundle(bundle_root)
    if bundle.get("manifest", {}).get("schema_version") != 3:
        raise ValueError("contextual extraction requires a schema-3 evaluation bundle")
    if _hash(exp_root / "manifest.json") != experiment_manifest_sha256:
        raise ValueError("experiment manifest pin mismatch")
    six_hashes = _exp_hashes(exp_root, experiment_manifest_sha256)
    if bundle["manifest"].get("source_experiment", {}).get("hashes") != {k: v for k, v in six_hashes.items() if k != "acoustic_targets.npy"}:
        raise ValueError("bundle and explicit experiment source inventory disagree")
    output = Path(output_root).resolve()
    entries, targets, input_roots = [], set(), {bundle_root, exp_root}
    for model in models:
        model_ids = run_ids[model]
        if not isinstance(model_ids, dict) or set(model_ids) != {0, 1, 2, 3, 4} or any(type(s) is not int for s in model_ids):
            raise ValueError(f"run_ids must use integer seed keys 0..4 for {model}")
        run_root = Path(canonical[model]).resolve(strict=True)
        input_roots.add(run_root)
        found = discover_seed_runs(run_root, model, (0, 1, 2, 3, 4))
        if found["experiment_hashes"] != six_hashes:
            raise ValueError(f"training source hashes mismatch for {model}: {run_root}")
        for rec in found["seeds"]:
            seed, out_id = rec["seed"], model_ids[rec["seed"]]
            if not _safe_id(out_id):
                raise ValueError(f"unsafe representation output ID for {model} seed {seed}: {out_id!r}")
            run_dir = output / "contextual-representations" / out_id
            data_dir = run_dir / "data"
            if run_dir in targets:
                raise ValueError(f"output ID collision for {model} seed {seed}: {run_dir}")
            targets.add(run_dir)
            entries.append({"model_id": model, "seed": seed, "checkpoint_path": rec["checkpoint_path"],
                            "checkpoint_sha256": rec["checkpoint_sha256"], "training_run_path": str(run_root),
                            "training_run_id": found["run_id"], "training_manifest_sha256": found["manifest_sha256"],
                            "output_run_id": out_id, "output_run_dir": str(run_dir), "output_dir": str(data_dir)})
    if any(p.exists() for p in targets):
        raise FileExistsError("one or more planned output run directories already exist")
    if any(t == i or t in i.parents or i in t.parents for t in targets for i in input_roots):
        raise ValueError("planned output overlaps bundle, experiment, or training run input")
    return {"schema_version": 1, "mode": "extract", "bundle_path": str(bundle_root),
            "bundle_manifest_sha256": bundle_manifest_sha256, "experiment_path": str(exp_root),
            "experiment_manifest_sha256": experiment_manifest_sha256, "experiment_hashes": six_hashes,
            "training_runs": {m: str(canonical[m]) for m in models}, "source_sha256": source_sha256,
            "output_root": str(output), "device": device, "batch_size": batch_size,
            "expected_seeds": [0, 1, 2, 3, 4], "inventory": entries}


def _revalidate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("mode") != "extract":
        raise ValueError("unsupported or malformed contextual extraction plan")
    ids: dict[str, dict[int, str]] = {}
    inventory = plan.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("plan inventory is empty")
    for item in inventory:
        if not isinstance(item, dict) or type(item.get("seed")) is not int or item["seed"] not in range(5):
            raise ValueError("plan entry has invalid seed or shape")
        model = item.get("model_id")
        if not isinstance(model, str) or canonical_model_id(model) != model:
            raise ValueError("plan entry model ID is not canonical")
        ids.setdefault(model, {})[item["seed"]] = item.get("output_run_id")
    regenerated = plan_contextual_extraction(
        bundle_path=plan["bundle_path"], bundle_manifest_sha256=plan["bundle_manifest_sha256"],
        experiment_path=plan["experiment_path"], experiment_manifest_sha256=plan["experiment_manifest_sha256"],
        training_runs=plan["training_runs"], run_ids=ids, output_root=plan["output_root"],
        device=plan["device"], batch_size=plan["batch_size"], expected_seeds=(0, 1, 2, 3, 4),
        source_sha256=plan["source_sha256"])
    if regenerated != plan:
        raise ValueError("extraction plan was modified or is stale; regenerate it")
    return regenerated


def execute_contextual_extraction(plan: dict[str, Any], *, extractor: Callable[..., Any] | None = None,
                                 context_factory: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Execute only a revalidated full plan and verify all immutable contextual exports."""
    plan = _revalidate_plan(plan)
    from .representations import load_contextual_representations
    if extractor is None:
        from .encoders import extract_bilstm
        extractor = extract_bilstm
    bundle = load_evaluation_bundle(plan["bundle_path"])
    expected_ids = sorted(row["occurrence_id"] for row in bundle["occurrences"])
    result: dict[str, dict[int, str]] = {}
    exports = []
    for item in plan["inventory"]:
        try:
            lifecycle = context_factory(item) if context_factory is not None else nullcontext()
            with lifecycle:
                extractor(bundle_path=plan["bundle_path"], experiment_path=plan["experiment_path"],
                          checkpoint_path=item["checkpoint_path"], output_dir=item["output_dir"], device=plan["device"],
                          batch_size=plan["batch_size"], contextual_only=True, expected_model_id=item["model_id"],
                          expected_seed=item["seed"], source_sha256=plan["source_sha256"])
                loaded = load_contextual_representations(item["output_dir"])
                prov = loaded["manifest"]["provenance"]
                if (prov["model_id"] != item["model_id"] or prov["seed"] != item["seed"]
                        or prov["checkpoint_sha256"] != item["checkpoint_sha256"]
                        or prov["bundle_manifest_sha256"] != plan["bundle_manifest_sha256"]
                        or prov["source_sha256"] != plan["source_sha256"]
                        or prov["experiment_hashes"] != plan["experiment_hashes"]
                        or prov["config_sha256"] != plan["experiment_hashes"]["config.json"]
                        or prov["vocab_sha256"] != plan["experiment_hashes"]["vocab.json"]
                        or loaded["occurrence_ids"] != expected_ids):
                    raise ValueError("export provenance or occurrence inventory mismatch")
                digest = _hash(Path(item["output_dir"]) / "manifest.json")
            result.setdefault(item["model_id"], {})[item["seed"]] = item["output_dir"]
            exports.append({"model_id": item["model_id"], "seed": item["seed"],
                            "representation_dir": item["output_dir"], "manifest_sha256": digest,
                            "manifest_identity": prov})
        except Exception as error:
            raise ValueError(f"extraction failed for {item['model_id']} seed {item['seed']} at {item['output_dir']}: {error}") from error
    return {"representation_dirs": result, "exports": exports}
