"""Storage, aggregation, and random baselines for aligned representation matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_matrix(ids: list[str], matrix: np.ndarray, *, require_nonzero: bool = True) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != len(ids) or len(ids) != len(set(ids)):
        raise ValueError("representation IDs and matrix shape/uniqueness disagree")
    if ids != sorted(ids):
        raise ValueError("representation IDs must be stably sorted")
    if not np.isfinite(values).all():
        raise ValueError("representation matrix contains non-finite values")
    if require_nonzero and np.any(np.linalg.norm(values.astype(np.float64), axis=1) <= 1e-12):
        raise ValueError("representation matrix contains a zero-norm row")
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
