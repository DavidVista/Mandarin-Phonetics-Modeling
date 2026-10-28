"""Portable dataset-only experiment bundles with no ML runtime dependency."""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np

from .artifacts import read_jsonl, safe_artifact, sha256_file, write_json
from .data_contract import sample_id
from .ids import IDS_RULES_VERSION, build_mapping
from .pinyin import (
    FINALS, INITIALS, PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, TONES,
    decompose, validate_canonical_label,
)

PREPARED_SHA256 = "c562722d2a866f4f1904c746c86c5863a310a66ee3875d86fc36f254fd2d7c10"
SCHEMA_VERSION = 3
MODES = ["syllable", "components", "acoustic", "components-acoustic-lambda-1", "components-acoustic-lambda-10"]
REQUIRED_FILES = {
    "samples.jsonl", "ids/mapping.json", "ids/vocabulary.json", "ids/dictionary.jsonl",
    "ids/COPYING", "ids/LGPL", "acoustic_targets.npy", "vocab.json", "splits.json", "config.json",
    "provenance/prepared_manifest.json", "provenance/processed_manifest.json",
    "provenance/processed_config.json", "provenance/acoustic_manifest.json",
    "provenance/sound_inventory.jsonl", "provenance/prepared_corpus.jsonl",
    "provenance/prepared_acoustic_targets.pkl",
}
_ARCH = {
    "embedding_dim": 64, "ids_bigru_hidden_per_direction": 32, "ids_bigru_layers": 1,
    "sentence_bilstm_hidden_per_direction": 64, "sentence_bilstm_layers": 1, "context_dim": 128,
    "classification_hidden_dim": 64, "classification_dropout": 0.5,
}
_TRAIN = {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 32,
          "epochs": 20, "scheduler": None, "early_stopping": False, "gradient_clipping": None,
          "cosine_epsilon": 1e-8}


def default_experiment_config(acoustic_dim=1024):
    """Return a fresh config with the agreed protocol and current input pin."""
    return {"schema_version": SCHEMA_VERSION, "prepared_manifest_sha256": PREPARED_SHA256,
            "split_ratios": [0.8, 0.1, 0.1], "split_seed": 0, "training_seeds": [0, 1, 2, 3, 4],
            "architecture": {**_ARCH, "acoustic_dim": acoustic_dim},
            "training": copy.deepcopy(_TRAIN), "model_ids": list(MODES)}


def _validate_config(value, acoustic_dim):
    expected = default_experiment_config(acoustic_dim)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("invalid experiment config keys")
    cfg = copy.deepcopy(value)
    def exact_type_value(actual, wanted):
        if type(actual) is not type(wanted): return False
        if isinstance(wanted, dict): return set(actual) == set(wanted) and all(exact_type_value(actual[k], wanted[k]) for k in wanted)
        if isinstance(wanted, list): return len(actual) == len(wanted) and all(exact_type_value(a, b) for a, b in zip(actual, wanted))
        return actual == wanted
    if (type(cfg["schema_version"]) is not int or cfg["schema_version"] != SCHEMA_VERSION or
        not exact_type_value(cfg["architecture"], expected["architecture"]) or
        not exact_type_value(cfg["training"], _TRAIN) or not exact_type_value(cfg["model_ids"], MODES)):
        raise ValueError("experiment config conflicts with the training protocol")
    pin = cfg["prepared_manifest_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("prepared_manifest_sha256 must be a lowercase SHA-256")
    ratios = cfg["split_ratios"]
    if (not isinstance(ratios, list) or len(ratios) != 3 or
        any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in ratios) or
        not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError("split_ratios must contain three finite positive values summing to one")
    if type(cfg["split_seed"]) is not int or cfg["split_seed"] != 0:
        raise ValueError("split_seed must be the fixed protocol value 0")
    seeds = cfg["training_seeds"]
    if (not isinstance(seeds, list) or len(seeds) != 5 or any(type(s) is not int for s in seeds)
        or seeds != [0, 1, 2, 3, 4]):
        raise ValueError("training_seeds must be the fixed protocol values [0, 1, 2, 3, 4]")
    return cfg


def _split_ids(ids, seed, ratios):
    values = sorted(ids)
    random.Random(seed).shuffle(values)
    n = len(values)
    train_end = int(n * ratios[0])
    validation_end = train_end + int(n * ratios[1])
    result = {"train": values[:train_end], "validation": values[train_end:validation_end], "test": values[validation_end:]}
    if any(not group for group in result.values()):
        raise ValueError("train, validation, and test splits must all be nonempty")
    return result


def _safe_id(value):
    return (isinstance(value, str) and bool(value) and value not in {".", ".."}
            and "/" not in value and "\\" not in value and Path(value).name == value
            and PureWindowsPath(value).name == value)


def prepare_experiment(experiment_id, *, prepared_root, output_root=None, config=None):
    """Create, validate and atomically publish a portable dataset bundle."""
    if not _safe_id(experiment_id):
        raise ValueError("experiment_id must be one safe path component")
    src = Path(prepared_root).resolve()
    output = Path(output_root).resolve() if output_root else src.parent.parent / "experiments"
    target = (output / experiment_id).resolve()
    if target.parent != output:
        raise ValueError("experiment path escapes output_root")
    if target.exists():
        raise FileExistsError(f"experiment already exists: {target}")

    from .preparation import validate_prepared
    prepared_manifest = validate_prepared(src)
    pin = sha256_file(src / "manifest.json")
    acoustic_bytes = (src / "acoustic/targets.pkl").read_bytes()
    if sha256_file(src / "acoustic/targets.pkl") != prepared_manifest["artifacts"]["acoustic/targets.pkl"]:
        raise ValueError("prepared acoustic target hash mismatch")
    acoustic = pickle.loads(acoustic_bytes)
    keys = sorted(acoustic)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("acoustic target inventory is empty or duplicated")
    matrix = np.stack([np.asarray(acoustic[k], dtype=np.float32) for k in keys])
    if (matrix.ndim != 2 or not np.isfinite(matrix).all() or
        np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) <= 1e-12)):
        raise ValueError("acoustic targets must be finite nonzero vectors")
    cfg = _validate_config(config if config is not None else default_experiment_config(matrix.shape[1]), matrix.shape[1])
    if pin != cfg["prepared_manifest_sha256"]:
        raise ValueError("prepared manifest checksum does not match config pin")

    rows = read_jsonl(src / "corpus.jsonl")
    key_ids = {label: i for i, label in enumerate(keys)}
    for row in rows:
        labels = row["labels"]
        component_ids = [list(decompose(validate_canonical_label(label))) for label in labels]
        row["component_labels"] = component_ids
        row["syllable_ids"] = [key_ids[label] for label in labels]
        row["acoustic_ids"] = list(row["syllable_ids"])

    output.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"experiment already exists: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{experiment_id}.", dir=output))
    try:
        (temp / "ids").mkdir()
        (temp / "provenance").mkdir()
        with (temp / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for name in ("mapping.json", "vocabulary.json", "dictionary.jsonl", "COPYING", "LGPL"):
            shutil.copyfile(src / "ids" / name, temp / "ids" / name)
        for source, relative in (
            (src / "manifest.json", "prepared_manifest.json"),
            (src / "reports/processed_corpus_manifest.json", "processed_manifest.json"),
            (src / "reports/processed_corpus_config.json", "processed_config.json"),
            (src / "reports/acoustic_bundle_manifest.json", "acoustic_manifest.json"),
            (src / "reports/sound_inventory.jsonl", "sound_inventory.jsonl"),
            (src / "corpus.jsonl", "prepared_corpus.jsonl"),
        ):
            shutil.copyfile(source, temp / "provenance" / relative)
        (temp / "provenance/prepared_acoustic_targets.pkl").write_bytes(acoustic_bytes)
        np.save(temp / "acoustic_targets.npy", matrix, allow_pickle=False)
        vocabulary = json.loads((src / "ids/vocabulary.json").read_text(encoding="utf-8"))
        mapping = json.loads((src / "ids/mapping.json").read_text(encoding="utf-8"))
        write_json(temp / "vocab.json", {"ids": vocabulary, "ids_mapping": mapping, "acoustic_tokens": keys,
                    "syllables": keys, "labels": {"initials": INITIALS, "finals": FINALS, "tones": TONES}})
        write_json(temp / "splits.json", _split_ids([row["sample_id"] for row in rows], cfg["split_seed"], cfg["split_ratios"]))
        write_json(temp / "config.json", cfg)
        artifacts = {str(p.relative_to(temp)).replace("\\", "/"): sha256_file(p)
                     for p in temp.rglob("*") if p.is_file()}
        provenance = {name: sha256_file(temp / name) for name in sorted(artifacts) if name.startswith("provenance/")}
        write_json(temp / "manifest.json", {
            "bundle_type": "experiment_dataset", "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
            "pinyin_contract": PINYIN_CONTRACT_VERSION, "parser_provenance": PARSER_PROVENANCE,
            "ids_rules": IDS_RULES_VERSION, "prepared_manifest_sha256": pin,
            "prepared_dataset_id": prepared_manifest["dataset_id"], "provenance": provenance,
            "counts": {"samples": len(rows), "positions": sum(row["sentence_length"] for row in rows),
                       "acoustic_targets": len(keys), "acoustic_dim": matrix.shape[1]}, "artifacts": artifacts,
        })
        validate_experiment(temp)
        os.rename(temp, target)
        return target
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _validate_semantics(root, manifest, config, rows, splits, vocab, matrix):
    if manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE or manifest.get("ids_rules") != IDS_RULES_VERSION:
        raise ValueError("unsupported experiment provenance contract")
    pin = manifest.get("prepared_manifest_sha256")
    if sha256_file(root / "provenance/prepared_manifest.json") != pin or pin != config["prepared_manifest_sha256"]:
        raise ValueError("prepared provenance pin mismatch")
    prepared = json.loads((root / "provenance/prepared_manifest.json").read_text(encoding="utf-8"))
    if prepared.get("dataset_id") != manifest.get("prepared_dataset_id"):
        raise ValueError("prepared dataset identity mismatch")
    processed_path = root / "provenance/processed_manifest.json"
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    acoustic_path = root / "provenance/acoustic_manifest.json"
    acoustic_manifest = json.loads(acoustic_path.read_text(encoding="utf-8"))
    inventory_path = root / "provenance/sound_inventory.jsonl"
    inventory = read_jsonl(inventory_path)
    inventory_hash = sha256_file(inventory_path)
    corpus_path = root / "provenance/prepared_corpus.jsonl"
    acoustic_targets_path = root / "provenance/prepared_acoustic_targets.pkl"
    if (prepared.get("artifacts", {}).get("corpus.jsonl") != sha256_file(corpus_path) or
        prepared.get("artifacts", {}).get("acoustic/targets.pkl") != sha256_file(acoustic_targets_path) or
        prepared.get("processed_corpus_manifest_sha256") != sha256_file(processed_path) or
        prepared.get("acoustic_bundle_manifest_sha256") != sha256_file(acoustic_path) or
        processed.get("sources", {}).get("sound_inventory_sha256") != inventory_hash or
        acoustic_manifest.get("inventory_sha256") != inventory_hash or
        processed.get("artifacts", {}).get("config.json") != sha256_file(root / "provenance/processed_config.json")):
        raise ValueError("copied upstream provenance does not agree")
    if prepared.get("artifacts", {}).get("ids/mapping.json") != sha256_file(root / "ids/mapping.json"):
        raise ValueError("prepared IDS mapping provenance mismatch")
    if prepared.get("artifacts", {}).get("ids/vocabulary.json") != sha256_file(root / "ids/vocabulary.json"):
        raise ValueError("prepared IDS vocabulary provenance mismatch")
    if prepared.get("artifacts", {}).get("ids/dictionary.jsonl") != sha256_file(root / "ids/dictionary.jsonl"):
        raise ValueError("prepared IDS dictionary provenance mismatch")
    for name, digest in manifest["provenance"].items():
        if sha256_file(safe_artifact(root, name)) != digest:
            raise ValueError(f"provenance checksum mismatch: {name}")
    if set(manifest["provenance"]) != {n for n in manifest["artifacts"] if n.startswith("provenance/")}:
        raise ValueError("provenance file set mismatch")

    original_vocab = json.loads((root / "ids/vocabulary.json").read_text(encoding="utf-8"))
    original_mapping = json.loads((root / "ids/mapping.json").read_text(encoding="utf-8"))
    bundled_vocab = vocab["ids"]
    if bundled_vocab != original_vocab or vocab["ids_mapping"] != original_mapping:
        raise ValueError("duplicated IDS mapping or vocabulary mismatch")
    if original_vocab.get("tokens", [None])[0] != "[PAD]" or original_vocab.get("stoi", {}).get("[PAD]") != 0:
        raise ValueError("IDS PAD token must have ID zero")
    ids_cfg = prepared.get("ids_provenance")
    rebuilt_mapping, rebuilt_vocab, _ = build_mapping(ids_cfg, dictionary_path=root / "ids/dictionary.jsonl")
    if rebuilt_mapping != original_mapping or rebuilt_vocab != original_vocab:
        raise ValueError("bundled IDS mapping or vocabulary fails reconstruction")

    tokens = vocab.get("acoustic_tokens")
    if (not isinstance(tokens, list) or not tokens or tokens != sorted(tokens) or len(tokens) != len(set(tokens))
        or vocab.get("syllables") != tokens or vocab.get("labels") != {"initials": INITIALS, "finals": FINALS, "tones": TONES}):
        raise ValueError("invalid acoustic or syllable vocabulary")
    inventory_labels = [row.get("label") for row in inventory]
    if sorted(inventory_labels) != tokens or len(inventory_labels) != len(set(inventory_labels)):
        raise ValueError("acoustic inventory does not match full syllable vocabulary")
    key_ids = {label: i for i, label in enumerate(tokens)}
    for label in tokens:
        validate_canonical_label(label)
    if matrix.shape != (len(tokens), config["architecture"]["acoustic_dim"]) or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix.astype(np.float64), axis=1) <= 1e-12):
        raise ValueError("invalid acoustic target matrix")
    with acoustic_targets_path.open("rb") as handle:
        pinned_acoustic = pickle.load(handle)
    pinned_keys = sorted(pinned_acoustic)
    pinned_matrix = np.stack([np.asarray(pinned_acoustic[k], dtype=np.float32) for k in pinned_keys])
    if pinned_keys != tokens or pinned_matrix.shape != matrix.shape or not np.array_equal(pinned_matrix, matrix):
        raise ValueError("acoustic matrix does not reproduce pinned prepared targets")
    mapping_chars = original_mapping.get("characters", {})
    id_count = len(original_vocab["tokens"])
    seen_ids = []
    seen_sources = set()
    for row in rows:
        required = {"component_lengths", "ids_sequences", "labels", "positions", "sample_id", "sentence_length", "source_ids", "text", "component_labels", "syllable_ids", "acoustic_ids"}
        if set(row) != required:
            raise ValueError("sample keys do not match experiment schema")
        text, labels = row["text"], row["labels"]
        n = len(text) if isinstance(text, str) else -1
        if n <= 0 or type(row["sentence_length"]) is not int or row["sentence_length"] != n:
            raise ValueError("invalid sample text or sentence length")
        if row["sample_id"] != sample_id(text, labels):
            raise ValueError("sample ID does not match text and labels")
        if (not isinstance(row["positions"], list) or any(type(pos) is not int for pos in row["positions"])
            or row["positions"] != list(range(n)) or not isinstance(labels, list) or len(labels) != n
            or not isinstance(row["ids_sequences"], list) or len(row["ids_sequences"]) != n
            or not isinstance(row["component_lengths"], list) or len(row["component_lengths"]) != n):
            raise ValueError("sample alignment mismatch")
        if (not isinstance(row["source_ids"], list) or not row["source_ids"] or
            any(not isinstance(s, str) or not s for s in row["source_ids"]) or len(set(row["source_ids"])) != len(row["source_ids"])):
            raise ValueError("invalid sample source IDs")
        if seen_sources.intersection(row["source_ids"]):
            raise ValueError("source IDs occur in more than one sample")
        seen_sources.update(row["source_ids"])
        expected_components, expected_syllables, expected_sequences = [], [], []
        for char, label, sequence, length in zip(text, labels, row["ids_sequences"], row["component_lengths"]):
            canonical = validate_canonical_label(label)
            expected_components.append(list(decompose(canonical)))
            if canonical not in key_ids:
                raise ValueError(f"missing acoustic target for label: {canonical}")
            expected_syllables.append(key_ids[canonical])
            if not isinstance(sequence, list) or not sequence or type(length) is not int or length != len(sequence):
                raise ValueError("invalid IDS sequence or component length")
            if any(type(token) is not int or token <= 0 or token >= id_count for token in sequence):
                raise ValueError("IDS token ID outside vocabulary or padding")
            char_entry = mapping_chars.get(char)
            if char_entry is None or sequence != char_entry.get("ids"):
                raise ValueError("IDS sequence does not match character mapping")
            expected_sequences.append(sequence)
        if (not isinstance(row["component_labels"], list) or
            any(not isinstance(x, list) or any(type(i) is not int for i in x) for x in row["component_labels"]) or
            row["component_labels"] != expected_components):
            raise ValueError("component targets mismatch canonical Pinyin")
        if any(type(i) is not int for i in row["syllable_ids"] + row["acoustic_ids"]):
            raise ValueError("syllable/acoustic IDs must be integers")
        if row["syllable_ids"] != expected_syllables or row["acoustic_ids"] != expected_syllables:
            raise ValueError("syllable/acoustic targets mismatch canonical Pinyin")
        seen_ids.append(row["sample_id"])
    if seen_ids != sorted(seen_ids) or len(seen_ids) != len(set(seen_ids)):
        raise ValueError("sample ID order or uniqueness invalid")
    source_rows = read_jsonl(corpus_path)
    base_fields = {"component_lengths", "ids_sequences", "labels", "positions", "sample_id", "sentence_length", "source_ids", "text"}
    projected_rows = [{key: row[key] for key in base_fields} for row in rows]
    if projected_rows != source_rows:
        raise ValueError("experiment samples do not match pinned prepared corpus")

    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("split names are invalid")
    split_ids = [sid for name in ("train", "validation", "test") for sid in splits[name]]
    if any(not splits[k] for k in splits) or len(split_ids) != len(set(split_ids)) or set(split_ids) != set(seen_ids):
        raise ValueError("split membership is incomplete or duplicated")
    if splits != _split_ids(seen_ids, config["split_seed"], config["split_ratios"]):
        raise ValueError("splits do not match deterministic config")
    counts = manifest.get("counts")
    expected_counts = {"samples": len(rows), "positions": sum(row["sentence_length"] for row in rows),
                       "acoustic_targets": len(tokens), "acoustic_dim": matrix.shape[1]}
    if counts != expected_counts:
        raise ValueError("experiment counts mismatch")


def validate_experiment(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"bundle_type", "schema_version", "experiment_id", "pinyin_contract", "parser_provenance", "ids_rules", "prepared_manifest_sha256", "prepared_dataset_id", "provenance", "counts", "artifacts"}:
        raise ValueError("invalid experiment manifest keys")
    if manifest.get("bundle_type") != "experiment_dataset" or manifest.get("schema_version") != SCHEMA_VERSION or not _safe_id(manifest.get("experiment_id")):
        raise ValueError("unsupported experiment bundle")
    artifacts = manifest["artifacts"]
    if set(artifacts) != REQUIRED_FILES:
        raise ValueError("experiment required artifact set mismatch")
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(artifacts):
        raise ValueError("experiment contains untracked or missing artifacts")
    for relative, digest in artifacts.items():
        p = safe_artifact(root, relative)
        if not p.is_file() or not isinstance(digest, str) or sha256_file(p) != digest:
            raise ValueError(f"artifact checksum mismatch: {relative}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    matrix = np.load(root / "acoustic_targets.npy", allow_pickle=False)
    config = _validate_config(config, matrix.shape[1] if matrix.ndim == 2 else -1)
    if manifest.get("prepared_manifest_sha256") != config["prepared_manifest_sha256"]:
        raise ValueError("manifest and config input pins disagree")
    vocabulary = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "samples.jsonl")
    _validate_semantics(root, manifest, config, rows, splits, vocabulary, matrix)
    return manifest


def load_experiment(path):
    """Load a portable bundle; samples are keyed by sample_id."""
    manifest = validate_experiment(path)
    root = Path(path).resolve()
    return {"root": root, "manifest": manifest, "config": json.loads((root / "config.json").read_text(encoding="utf-8")),
            "vocab": json.loads((root / "vocab.json").read_text(encoding="utf-8")),
            "splits": json.loads((root / "splits.json").read_text(encoding="utf-8")),
            "samples": {row["sample_id"]: row for row in read_jsonl(root / "samples.jsonl")},
            "acoustic_targets": np.load(root / "acoustic_targets.npy", allow_pickle=False)}
