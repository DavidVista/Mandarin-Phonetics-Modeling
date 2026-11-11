"""Pinned character-encoder inputs and pre-head occurrence exports."""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..artifacts import safe_artifact, sha256_file, write_json
from ..bundles import seal_bundle, staging_directory, validate_bundle
from ..experiment import load_common_test
from ..objectives import parse_model_id
from .geometry_bundle import validate_geometry_bundle
from .training_summary import TRAINING_MODELS, TRAINING_SEEDS, SELECTION_RULE

EXTRACTION_TYPE = "encoder_prehead_export"
SEED_TYPE = "encoder_prehead_seed"


def pinned_bytes(root, name, expected):
    data = safe_artifact(root, name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"checksum mismatch: {name}")
    return data


def load_positions(root, pin):
    manifest = validate_geometry_bundle(root)
    if json.loads(pinned_bytes(root, "manifest.json", pin)) != manifest:
        raise ValueError("evaluation manifest changed during validation")
    rows = [json.loads(line) for line in pinned_bytes(
        root, "occurrences.jsonl", manifest["artifacts"]["occurrences.jsonl"]
    ).splitlines()]
    rows.sort(key=lambda row: row["occurrence_id"])
    ids = [row["occurrence_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("evaluation occurrence identities are empty or duplicated")
    return manifest, rows


def inspect_checkpoints(root, pin, model, experiment_pin):
    """Check checkpoint bytes and selected epochs without loading tensors."""
    if model not in TRAINING_MODELS:
        raise ValueError("unknown encoder model")
    manifest = json.loads(pinned_bytes(root, "manifest.json", pin))
    if (manifest.get("run_type") != "training_run" or manifest.get("schema_version") != 1
            or manifest.get("config", {}).get("schema_version") != 4
            or manifest.get("model_id") != model or manifest.get("status") != "complete"
            or manifest.get("seeds") != list(TRAINING_SEEDS)
            or manifest.get("seed_status") != {str(seed): "complete" for seed in TRAINING_SEEDS}
            or manifest.get("selection") != SELECTION_RULE
            or manifest.get("experiment_hashes", {}).get("manifest.json") != experiment_pin):
        raise ValueError(f"incompatible character checkpoint manifest for {model}")
    model_source = (Path(__file__).parents[1] / "model.py").read_text(encoding="utf-8")
    source_pins = {hashlib.sha256(text.encode()).hexdigest()
                   for text in (model_source, model_source.replace("\n", "\r\n"))}
    if manifest.get("code_sha256", {}).get("model.py") not in source_pins:
        raise ValueError(f"trained encoder source differs for {model}")
    artifacts = manifest["artifacts"]
    selections = {}
    for seed in TRAINING_SEEDS:
        prefix = f"seed-{seed}"
        selection = json.loads(pinned_bytes(root, f"{prefix}/summary.json", artifacts[f"{prefix}/summary.json"]))
        history = json.loads(pinned_bytes(root, f"{prefix}/history.json", artifacts[f"{prefix}/history.json"]))["epochs"]
        best = min(history, key=lambda row: (row["validation"]["objective"], row["epoch"]))
        if (selection.get("seed") != seed or selection.get("status") != "complete"
                or selection.get("best_epoch") != best["epoch"]
                or selection.get("best_validation_metrics") != best["validation"]
                or selection.get("best_validation_objective") != best["validation"]["objective"]):
            raise ValueError(f"invalid selected checkpoint for {model}/{seed}")
        name = f"{prefix}/best.pt"
        if sha256_file(safe_artifact(root, name)) != artifacts[name]:
            raise ValueError(f"checkpoint checksum mismatch for {model}/{seed}")
        selections[seed] = selection
    return manifest, selections


def extract_hidden_states(*, evaluation_root, evaluation_pin, experiment_root, experiment_pin,
                          training_root, training_pin, model, output, device, batch_size,
                          source_sha256, allow_execution=False):
    """Encode full test sentences, retaining only the sealed occurrence positions."""
    if not allow_execution:
        raise PermissionError("encoder extraction requires explicit execution authorization")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    evaluation, rows = load_positions(evaluation_root, evaluation_pin)
    if evaluation["identity"]["experiment_manifest_sha256"] != experiment_pin:
        raise ValueError("evaluation and experiment pins differ")
    training, selections = inspect_checkpoints(training_root, training_pin, model, experiment_pin)
    test_ids, samples = load_common_test(experiment_root, expected_manifest_sha256=experiment_pin)
    from ..bundles import content_hash
    if content_hash(test_ids) != evaluation["identity"]["test_ids_sha256"]:
        raise ValueError("evaluation and experiment test identities differ")
    experiment = json.loads(pinned_bytes(experiment_root, "manifest.json", experiment_pin))
    config = json.loads(pinned_bytes(experiment_root, "config.json", experiment["artifacts"]["config.json"]))
    labels = json.loads(pinned_bytes(experiment_root, "labels.json", experiment["artifacts"]["labels.json"]))
    if config != training["config"]:
        raise ValueError("training and experiment configurations differ")
    sample_map = {sample["sample_id"]: sample for sample in samples}
    by_sample = defaultdict(list)
    for index, row in enumerate(rows):
        sample = sample_map[row["sample_id"]]
        position = row["character_position"]
        if (type(position) is not int or not 0 <= position < len(sample["text"])
                or sample["text"][position] != row["character"] or sample["labels"][position] != row["pinyin"]
                or row["occurrence_id"] != f"{row['sample_id']}:{position}"):
            raise ValueError("evaluation occurrence does not match source test position")
        by_sample[row["sample_id"]].append((index, position))

    import torch
    from ..model import build_encoder
    from ..pinyin import FINALS, INITIALS, TONES

    if torch.__version__.split("+", 1)[0] != "2.11.0":
        raise RuntimeError("encoder extraction requires Torch 2.11.0")
    target = Path(output).resolve()
    for root in (evaluation_root, experiment_root, training_root):
        if target.is_relative_to(Path(root).resolve()) or Path(root).resolve().is_relative_to(target):
            raise ValueError("extraction output must be separate from inputs")
    identity = {"model": model, "evaluation_manifest_sha256": evaluation_pin,
                "experiment_manifest_sha256": experiment_pin, "training_manifest_sha256": training_pin,
                "seeds": list(TRAINING_SEEDS), "source_sha256": source_sha256,
                "device": device, "batch_size": batch_size, "torch": torch.__version__}
    with staging_directory(target) as stage:
        for seed, selected in selections.items():
            base = f"seeds/{seed}"
            artifacts = experiment["artifacts"]
            vocab_bytes = pinned_bytes(experiment_root, f"{base}/vocabulary.json", artifacts[f"{base}/vocabulary.json"])
            vocabulary = json.loads(vocab_bytes)
            seed_hashes = json.loads(pinned_bytes(
                experiment_root, f"{base}/input_hashes.json", artifacts[f"{base}/input_hashes.json"]
            ))
            if seed_hashes != selected["seed_input_hashes"] or seed_hashes["test"] != evaluation["identity"]["test_ids_sha256"]:
                raise ValueError("seed vocabulary or split identity differs")
            name = f"seed-{seed}/best.pt"
            checkpoint_sha = training["artifacts"][name]
            checkpoint = torch.load(io.BytesIO(pinned_bytes(training_root, name, checkpoint_sha)),
                                    map_location="cpu", weights_only=True)
            expected = {
                "seed": seed, "model_id": model, "lambda_cosine": parse_model_id(model)[1],
                "best_epoch": selected["best_epoch"], "validation_objective": selected["best_validation_objective"],
                "seed_input_hashes": seed_hashes, "experiment_hashes": training["experiment_hashes"],
                "config_sha256": artifacts["config.json"], "vocab_sha256": artifacts[f"{base}/vocabulary.json"],
            }
            if any(checkpoint.get(key) != value for key, value in expected.items()):
                raise ValueError(f"checkpoint payload identity differs for seed {seed}")
            vocab = {"characters": vocabulary, "syllables": labels,
                     "labels": {"initials": INITIALS, "finals": FINALS, "tones": TONES}}
            encoder = build_encoder(config, vocab, model_id=model, seed=seed)
            encoder.load_state_dict(checkpoint["state_dict"], strict=True)
            encoder.to(device).eval()
            hidden = np.empty((len(rows), encoder.context_dim), dtype=np.float32)
            unknown = np.zeros(len(rows), dtype=bool)
            with torch.inference_mode():
                for start in range(0, len(samples), batch_size):
                    batch = samples[start:start+batch_size]
                    tokens = [[vocabulary["stoi"].get(char, 1) for char in sample["text"]] for sample in batch]
                    lengths = torch.tensor([len(row) for row in tokens])
                    ids = torch.zeros((len(tokens), int(lengths.max())), dtype=torch.long, device=device)
                    for i, token_row in enumerate(tokens):
                        ids[i, :len(token_row)] = torch.tensor(token_row, device=device)
                    mask = ids != 0
                    values, result_mask = encoder.encode({"token_ids": ids, "sentence_lengths": lengths, "char_mask": mask})
                    if not torch.equal(mask, result_mask):
                        raise ValueError("encoder returned a different position mask")
                    values = values.detach().cpu().numpy()
                    for i, sample in enumerate(batch):
                        for index, position in by_sample[sample["sample_id"]]:
                            hidden[index] = values[i, position]
                            unknown[index] = tokens[i][position] == 1
            if not np.isfinite(hidden).all() or np.any(np.linalg.norm(hidden, axis=1) == 0):
                raise ValueError(f"nonfinite or zero hidden state in seed {seed}")
            root = stage / f"seed-{seed}"
            root.mkdir()
            write_json(root / "occurrence_ids.json", [row["occurrence_id"] for row in rows])
            np.save(root / "hidden.npy", hidden, allow_pickle=False)
            np.save(root / "unknown.npy", unknown, allow_pickle=False)
            seal_bundle(root, SEED_TYPE, {**identity, "seed": seed, "checkpoint_sha256": checkpoint_sha,
                         "best_epoch": selected["best_epoch"], "seed_input_hashes": seed_hashes,
                         "representation": "raw pre-head"})
        seal_bundle(stage, EXTRACTION_TYPE, identity)
    return target


def load_export(root, *, model, evaluation_pin, experiment_pin, occurrence_ids, training_pin=None,
                expected_seeds=None):
    manifest = validate_bundle(root, EXTRACTION_TYPE)
    identity = manifest["identity"]
    if (identity.get("model") != model or identity.get("evaluation_manifest_sha256") != evaluation_pin
            or identity.get("experiment_manifest_sha256") != experiment_pin
            or identity.get("seeds") != list(TRAINING_SEEDS)
            or (training_pin is not None and identity.get("training_manifest_sha256") != training_pin)):
        raise ValueError(f"hidden-state export identity mismatch for {model}")
    expected = {f"seed-{seed}/{name}" for seed in TRAINING_SEEDS
                for name in ("manifest.json", "occurrence_ids.json", "hidden.npy", "unknown.npy")}
    if set(manifest["artifacts"]) != expected:
        raise ValueError(f"hidden-state export artifact set differs for {model}")
    for seed in TRAINING_SEEDS:
        seed_root = Path(root) / f"seed-{seed}"
        child = validate_bundle(seed_root, SEED_TYPE)
        child_pin = manifest["artifacts"][f"seed-{seed}/manifest.json"]
        if json.loads(pinned_bytes(seed_root, "manifest.json", child_pin)) != child:
            raise ValueError(f"hidden-state seed manifest changed for {model}/{seed}")
        info = child["identity"]
        if info.get("seed") != seed or any(info.get(key) != value for key, value in identity.items()):
            raise ValueError(f"hidden-state seed identity mismatch for {model}/{seed}")
        if expected_seeds is not None and any(info.get(key) != value for key, value in expected_seeds[seed].items()):
            raise ValueError(f"hidden-state checkpoint provenance mismatch for {model}/{seed}")
        seed_hashes = info.get("seed_input_hashes")
        if (info.get("representation") != "raw pre-head" or type(info.get("best_epoch")) is not int
                or info["best_epoch"] < 1 or not re.fullmatch("[0-9a-f]{64}", str(info.get("checkpoint_sha256")))
                or not isinstance(seed_hashes, dict) or set(seed_hashes) != {"train", "validation", "test", "vocabulary"}
                or any(not re.fullmatch("[0-9a-f]{64}", str(value)) for value in seed_hashes.values())):
            raise ValueError(f"invalid hidden-state provenance for {model}/{seed}")
        def read(name):
            return pinned_bytes(root, f"seed-{seed}/{name}", manifest["artifacts"][f"seed-{seed}/{name}"])
        ids = json.loads(read("occurrence_ids.json"))
        hidden = np.load(io.BytesIO(read("hidden.npy")), allow_pickle=False)
        unknown = np.load(io.BytesIO(read("unknown.npy")), allow_pickle=False)
        if (ids != occurrence_ids or hidden.ndim != 2 or hidden.shape[0] != len(ids)
                or hidden.shape[1] < 1 or hidden.dtype != np.float32 or not np.isfinite(hidden).all()
                or np.any(np.linalg.norm(hidden.astype(np.float64), axis=1) == 0)
                or unknown.shape != (len(ids),) or unknown.dtype != np.bool_):
            raise ValueError(f"hidden-state occurrence alignment differs for {model}/{seed}")
        yield seed, hidden, unknown, info
