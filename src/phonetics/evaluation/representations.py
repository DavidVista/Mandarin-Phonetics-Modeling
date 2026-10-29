"""Storage, aggregation, and random baselines for aligned representation matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ..objectives import canonical_model_id


_SOURCE_HASH_NAMES = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"}
_PROVENANCE_HASH_NAMES = ("checkpoint_sha256", "bundle_manifest_sha256", "config_sha256", "vocab_sha256", "source_sha256")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_contextual_provenance(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("contextual representation provenance is required")
    required = {"model_id", "seed", *_PROVENANCE_HASH_NAMES, "experiment_hashes", "environment", "evaluation_mode"}
    if not required.issubset(provenance):
        raise ValueError("contextual provenance is incomplete")
    try:
        model_id = canonical_model_id(provenance["model_id"])
    except (TypeError, ValueError) as error:
        raise ValueError("contextual provenance model_id is invalid") from error
    if model_id != provenance["model_id"]:
        raise ValueError("contextual provenance model_id is not canonical")
    if not isinstance(provenance["seed"], int) or isinstance(provenance["seed"], bool):
        raise ValueError("contextual seed must be an integer")
    for name in _PROVENANCE_HASH_NAMES:
        if not _is_sha256(provenance[name]):
            raise ValueError(f"contextual provenance hash is invalid: {name}")
    hashes = provenance["experiment_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != _SOURCE_HASH_NAMES or any(not _is_sha256(v) for v in hashes.values()):
        raise ValueError("contextual experiment hash inventory is invalid")
    if hashes["config.json"] != provenance["config_sha256"] or hashes["vocab.json"] != provenance["vocab_sha256"]:
        raise ValueError("contextual config/vocabulary hashes disagree with experiment hashes")
    environment = provenance["environment"]
    env_fields = {"python", "numpy", "torch", "platform", "device"}
    if not isinstance(environment, dict) or set(environment) != env_fields or any(
            not isinstance(environment[key], str) or not environment[key].strip() for key in env_fields):
        raise ValueError("contextual environment provenance is invalid")
    if provenance["evaluation_mode"] is not True:
        raise ValueError("contextual extraction must be verified in evaluation mode")
    return provenance


def validate_matrix(ids: list[str], matrix: np.ndarray, *, require_nonzero: bool = True) -> np.ndarray:
    values = np.asarray(matrix)
    if (not isinstance(ids, list) or not ids or values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] == 0
            or any(not isinstance(x, str) or not x for x in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("representation IDs and matrix shape/uniqueness disagree")
    if values.dtype.kind != "f":
        raise ValueError("representation matrix must use a real floating-point dtype")
    if ids != sorted(ids):
        raise ValueError("representation IDs must be stably sorted")
    finite_rows = np.isfinite(values).all(axis=1)
    if not finite_rows.all():
        raise ValueError(f"representation matrix contains non-finite occurrences: {[ids[i] for i in np.flatnonzero(~finite_rows)]}")
    if require_nonzero:
        # Scale before squaring: direct norm may underflow for tiny values or overflow
        # for large, still-finite values. Only an exactly zero row is invalid.
        rows = values.astype(np.float64)
        scales = np.max(np.abs(rows), axis=1)
        bad = np.flatnonzero(scales == 0)
        if len(bad):
            raise ValueError(f"representation matrix contains zero-norm occurrences: {[ids[i] for i in bad]}")
    return values


def aggregate_occurrences(
    occurrence_ids: list[str], hidden: np.ndarray, groups: list[dict[str, Any]], group_kind: str,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Mean raw hidden states over every occurrence member of each group."""
    values = validate_matrix(occurrence_ids, hidden)
    positions = {occurrence_id: index for index, occurrence_id in enumerate(occurrence_ids)}
    selected = sorted((row for row in groups if row.get("group_kind") == group_kind), key=lambda row: row["group_id"])
    group_ids, means, counts = [], [], []
    for group in selected:
        members = group.get("occurrence_ids")
        if not isinstance(members, list) or not members or len(members) != len(set(members)):
            raise ValueError(f"invalid membership for group {group.get('group_id')}")
        try:
            indices = [positions[member] for member in members]
        except KeyError as error:
            raise ValueError(f"group references missing occurrence {error.args[0]}") from error
        group_ids.append(group["group_id"])
        means.append(values[indices].astype(np.float64).mean(axis=0))
        counts.append(len(indices))
    matrix = np.asarray(means, dtype=np.float32)
    validate_matrix(group_ids, matrix)
    return group_ids, matrix, counts


def random_occurrence_baseline(occurrence_ids: list[str], *, dimension: int, seed: int) -> np.ndarray:
    if dimension <= 0 or isinstance(dimension, bool):
        raise ValueError("dimension must be a positive integer")
    if occurrence_ids != sorted(occurrence_ids) or len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("occurrence IDs must be unique and sorted")
    matrix = np.random.default_rng(seed).standard_normal((len(occurrence_ids), dimension)).astype(np.float32)
    return validate_matrix(occurrence_ids, matrix)


def save_representations(
    output_dir: str | Path, *, occurrence_ids: list[str], hidden: np.ndarray,
    groups: list[dict[str, Any]], provenance: dict[str, Any],
) -> Path:
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"representation output already exists: {target}")
    validate_matrix(occurrence_ids, hidden)
    aggregates = {
        kind: aggregate_occurrences(occurrence_ids, hidden, groups, kind)
        for kind in ("character_pinyin", "syllable")
    }
    target.mkdir(parents=True)
    (target / "occurrence_ids.json").write_text(json.dumps(occurrence_ids, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(target / "occurrence_hidden.npy", np.asarray(hidden, dtype=np.float32), allow_pickle=False)
    for kind, (ids, matrix, counts) in aggregates.items():
        (target / f"{kind}_ids.json").write_text(json.dumps(ids, ensure_ascii=False) + "\n", encoding="utf-8")
        (target / f"{kind}_counts.json").write_text(json.dumps(counts) + "\n", encoding="utf-8")
        np.save(target / f"{kind}_hidden.npy", matrix, allow_pickle=False)
    artifacts = {}
    for path in target.iterdir():
        if path.is_file():
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "provenance": provenance, "artifacts": artifacts}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def save_contextual_representations(
    output_dir: str | Path, *, occurrence_ids: list[str], hidden: np.ndarray,
    provenance: dict[str, Any],
) -> Path:
    """Persist raw contextual h only; the output is immutable and hash pinned."""
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"representation output already exists: {target}")
    values = validate_matrix(occurrence_ids, hidden)
    provenance = _validate_contextual_provenance(provenance)
    target.mkdir(parents=True)
    (target / "occurrence_ids.json").write_text(json.dumps(occurrence_ids, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(target / "occurrence_hidden.npy", values, allow_pickle=False)
    artifacts = {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                 for name in ("occurrence_ids.json", "occurrence_hidden.npy")}
    manifest = {"schema_version": 1, "representation": "contextual_hidden_prehead", "provenance": provenance,
                "dimension": int(values.shape[1]), "dtype": str(values.dtype), "evaluation_mode": True,
                "normalization": "none", "artifacts": artifacts}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def load_contextual_representations(path: str | Path) -> dict[str, Any]:
    """Load and verify an immutable raw contextual representation run."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("representation") != "contextual_hidden_prehead":
        raise ValueError("unsupported contextual representation manifest")
    expected_artifacts = {"occurrence_ids.json", "occurrence_hidden.npy"}
    artifact_inventory = manifest.get("artifacts")
    if not isinstance(artifact_inventory, dict) or set(artifact_inventory) != expected_artifacts:
        raise ValueError("contextual artifact inventory is invalid")
    for name, digest in artifact_inventory.items():
        if not _is_sha256(digest):
            raise ValueError(f"contextual artifact checksum is invalid: {name}")
        if _file_sha256(root / name) != digest:
            raise ValueError(f"contextual artifact checksum mismatch: {name}")
    provenance = manifest.get("provenance", {})
    _validate_contextual_provenance(provenance)
    if (manifest.get("evaluation_mode") is not True or provenance.get("evaluation_mode") is not manifest.get("evaluation_mode")
            or manifest.get("normalization") != "none"):
        raise ValueError("contextual representation mode or normalization is invalid")
    ids = json.loads((root / "occurrence_ids.json").read_text(encoding="utf-8"))
    hidden = np.load(root / "occurrence_hidden.npy", allow_pickle=False)
    validate_matrix(ids, hidden)
    dimension = manifest.get("dimension")
    if (not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
            or hidden.shape[1] != dimension or str(hidden.dtype) != manifest.get("dtype")):
        raise ValueError("contextual artifact shape or dtype disagrees with manifest")
    return {"occurrence_ids": ids, "hidden": hidden, "manifest": manifest}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
