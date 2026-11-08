"""Inventory-bound acoustic targets and character prepared datasets."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil

from .artifacts import read_jsonl, safe_artifact, sha256_file, write_json, write_jsonl
from .bundles import seal_bundle, staging_directory, validate_bundle
from .corpus import validate_character_corpus, validate_corpus_records
from .data_contract import validate_acoustic
from .pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, validate_canonical_label
from .sound_inventory import validate_sound_inventory


def inventory_signature(rows):
    return sorted((row["label"], row["sha256"]) for row in rows)


def prepare_acoustic_targets(inventory_path, destination, *, model, cache_root=None,
                             existing_bundle=None, existing_manifest_sha256=None, backend=None):
    """Compute targets or import a pinned historical acoustic-only bundle.

    Import accepts only the exact inventory/audio identity. Historical corpus,
    IDS, sampling and prepared schemas are never consumed by this operation.
    """
    import numpy as np
    from .preparation.config import validate_model_config

    model = validate_model_config(model)
    if len(model["revision"]) != 40 or any(c not in "0123456789abcdef" for c in model["revision"]):
        raise ValueError("model revision must be an exact 40-character commit SHA")
    inventory_path = Path(inventory_path)
    validate_sound_inventory(inventory_path)
    rows = read_jsonl(inventory_path / "inventory.jsonl")
    identity = {"inventory_manifest_sha256": sha256_file(inventory_path / "manifest.json"),
                "model": model, "contract": "acoustic-targets-1", "backend": "fixture" if backend else "transformers"}
    if existing_bundle is not None:
        if backend is not None or existing_manifest_sha256 is None:
            raise ValueError("existing acoustic bundle requires an explicit pin and no inference backend")
        root = Path(existing_bundle)
        if sha256_file(root / "manifest.json") != existing_manifest_sha256:
            raise ValueError("acoustic input manifest checksum mismatch")
        old = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if old.get("bundle_type") != "acoustic_embeddings" or old.get("schema_version") != 1:
            raise ValueError("unsupported acoustic input schema")
        required = {"pinyin_embeddings.pkl", "sound_inventory.jsonl", "config.json", "reports/inference.json"}
        if not required.issubset(old.get("artifacts", {})):
            raise ValueError("acoustic input is missing required authenticated artifacts")
        if old.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or old.get("parser_provenance") != PARSER_PROVENANCE:
            raise ValueError("acoustic input Pinyin provenance mismatch")
        for name, digest in old["artifacts"].items():
            if sha256_file(safe_artifact(root, name)) != digest:
                raise ValueError("acoustic input artifact checksum mismatch")
        old_rows = read_jsonl(root / "sound_inventory.jsonl")
        if old.get("inventory_sha256") != sha256_file(root / "sound_inventory.jsonl"):
            raise ValueError("acoustic input inventory checksum mismatch")
        old_labels = [validate_canonical_label(r["label"]) for r in old_rows]
        if len(set(old_labels)) != len(old_labels) or not old_labels:
            raise ValueError("acoustic input inventory labels invalid")
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        inference = json.loads((root / "reports/inference.json").read_text(encoding="utf-8"))
        if (cfg.get("bundle_id") != old.get("bundle_id")
                or {**validate_model_config(cfg["model"]), "backend": inference.get("backend")} != old.get("model")
                or inference != old.get("inference") or inference.get("backend") != "transformers"
                or any(type(inference.get(k)) is not int or inference[k] < 0 for k in ("computed", "cached", "errors"))
                or inference["errors"] != 0 or inference["computed"] + inference["cached"] != len(old_rows)
                or type(old.get("count")) is not int or old["count"] != len(old_rows)
                or type(old.get("dimension")) is not int or old["dimension"] <= 0):
            raise ValueError("acoustic input inference/model metadata mismatch")
        if inventory_signature(old_rows) != inventory_signature(rows):
            raise ValueError("acoustic input inventory/audio mismatch")
        old_model = dict(old["model"])
        old_model.pop("backend", None)
        # Device is execution provenance, not a difference in target semantics.
        if {k: v for k, v in old_model.items() if k != "device"} != {k: v for k, v in model.items() if k != "device"}:
            raise ValueError("acoustic model parameters mismatch")
        identity.update({"source_manifest_sha256": existing_manifest_sha256, "backend": "imported"})
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        validate_acoustic_targets(target, identity=identity)
        return target
    if existing_bundle is not None:
        with (root / "pinyin_embeddings.pkl").open("rb") as handle:
            vectors = pickle.load(handle)
        report = {"source_manifest_sha256": existing_manifest_sha256, "computed": 0, "imported": len(vectors)}
    else:
        if cache_root is None:
            raise ValueError("cache_root is required for acoustic inference")
        from .preparation.embeddings import embed_inventory
        runtime_rows = [{**row, "path": str(safe_artifact(inventory_path, row["relative_path"]))} for row in rows]
        vectors, report = embed_inventory(runtime_rows, model, cache_root, backend)
    validate_acoustic(vectors, expected_keys=[r["label"] for r in rows])
    if existing_bundle is not None and validate_acoustic(vectors) != old["dimension"]:
        raise ValueError("acoustic input target dimension mismatch")
    with staging_directory(target) as temporary:
        labels = sorted(vectors)
        np.save(temporary / "targets.npy", np.stack([vectors[k] for k in labels]).astype(np.float32), allow_pickle=False)
        write_json(temporary / "labels.json", labels)
        write_jsonl(temporary / "inventory.jsonl", rows)
        write_json(temporary / "inference.json", report)
        seal_bundle(temporary, "acoustic_targets", identity)
        validate_acoustic_targets(temporary)
    return target


def validate_acoustic_targets(path, *, identity=None):
    root = Path(path)
    manifest = validate_bundle(root, "acoustic_targets", identity=identity)
    validate_target_records(root)
    return manifest


def validate_target_records(root):
    import numpy as np
    root = Path(root)
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    if not labels or labels != sorted(set(labels)):
        raise ValueError("acoustic labels must be sorted and unique")
    for label in labels:
        validate_canonical_label(label)
    matrix = np.load(root / "targets.npy", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or matrix.dtype != np.float32:
        raise ValueError("invalid acoustic target matrix")
    validate_acoustic(dict(zip(labels, matrix)), expected_keys=[r["label"] for r in read_jsonl(root / "inventory.jsonl")])


def assemble_character_data(corpus_path, acoustic_path, destination):
    corpus_path, acoustic_path = Path(corpus_path), Path(acoustic_path)
    cm = validate_character_corpus(corpus_path)
    am = validate_acoustic_targets(acoustic_path)
    if cm["identity"]["inventory_manifest_sha256"] != am["identity"]["inventory_manifest_sha256"]:
        raise ValueError("corpus and acoustic inventory provenance mismatch")
    if inventory_signature(read_jsonl(corpus_path / "inventory.jsonl")) != inventory_signature(read_jsonl(acoustic_path / "inventory.jsonl")):
        raise ValueError("corpus and acoustic inventory mismatch")
    identity = {"corpus_manifest_sha256": sha256_file(corpus_path / "manifest.json"),
                "acoustic_manifest_sha256": sha256_file(acoustic_path / "manifest.json"),
                "contract": "character-prepared-1"}
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        validate_character_data(target, identity=identity)
        return target
    with staging_directory(target) as temporary:
        for name in ("corpus.jsonl", "statistics.json", "rejections.jsonl", "inventory.jsonl"):
            shutil.copyfile(corpus_path / name, temporary / name)
        for name in ("labels.json", "targets.npy", "inference.json"):
            shutil.copyfile(acoustic_path / name, temporary / name)
        write_json(temporary / "corpus_manifest.json", cm)
        write_json(temporary / "acoustic_manifest.json", am)
        seal_bundle(temporary, "character_prepared", identity)
        validate_character_data(temporary)
    return target


def validate_character_data(path, *, identity=None):
    root = Path(path)
    manifest = validate_bundle(root, "character_prepared", identity=identity)
    for name, field in (("corpus_manifest.json", "corpus_manifest_sha256"),
                        ("acoustic_manifest.json", "acoustic_manifest_sha256")):
        if sha256_file(root / name) != manifest["identity"][field]:
            raise ValueError("prepared provenance checksum mismatch")
    cm = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    am = json.loads((root / "acoustic_manifest.json").read_text(encoding="utf-8"))
    from .bundles import content_hash
    for upstream, kind, names in (
        (cm, "character_corpus", ("corpus.jsonl", "statistics.json", "rejections.jsonl", "inventory.jsonl")),
        (am, "acoustic_targets", ("targets.npy", "labels.json", "inference.json", "inventory.jsonl"))):
        if (upstream.get("bundle_type") != kind or upstream.get("schema_version") != 1
                or upstream.get("identity_sha256") != content_hash(upstream.get("identity"))
                or set(upstream.get("artifacts", {})) != set(names)):
            raise ValueError("prepared upstream contract mismatch")
        for name in names:
            if sha256_file(root / name) != upstream["artifacts"][name]:
                raise ValueError(f"prepared upstream artifact mismatch: {name}")
    if cm["identity"]["inventory_manifest_sha256"] != am["identity"]["inventory_manifest_sha256"]:
        raise ValueError("prepared upstream inventory provenance mismatch")
    validate_corpus_records(root)
    validate_target_records(root)
    return manifest
