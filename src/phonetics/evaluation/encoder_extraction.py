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
from ..bundles import content_hash, seal_bundle, staging_directory, validate_bundle
from ..architecture import BERT_REQUIREMENTS
from ..experiment import SCHEMA_VERSION, SELECTION_RULE, load_common_test
from ..objectives import canonical_model_id, is_extension_model, parse_model_id
from ..bert import is_bert
from ..bert_snapshot import load_bert_inputs
from ..training_artifacts import TRAINING_RUN_SCHEMAS
from .geometry_bundle import validate_geometry_bundle

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


def load_experiment_manifest(root, pin):
    manifest = json.loads(pinned_bytes(root, "manifest.json", pin))
    identity = manifest["identity"]
    config = identity["config"]
    models, seeds = config["model_ids"], identity["training_seeds"]
    if (manifest.get("bundle_type") != "character_experiment"
            or content_hash(identity) != manifest.get("identity_sha256")
            or config.get("schema_version") != SCHEMA_VERSION
            or not isinstance(models, list) or not models
            or [canonical_model_id(model) for model in models] != models or len(set(models)) != len(models)
            or type(config.get("seed_count")) is not int or config["seed_count"] < 1
            or not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds)
            or seeds != list(range(config["seed_count"]))):
        raise ValueError("invalid encoder experiment model or seed inventory")
    return manifest


def inspect_checkpoints(root, pin, model, experiment_pin, experiment):
    """Check checkpoint bytes and selected epochs without loading tensors."""
    config = experiment["identity"]["config"]
    seeds = experiment["identity"]["training_seeds"]
    if model not in config["model_ids"] and not is_extension_model(model):
        raise ValueError("unknown encoder model")
    manifest = json.loads(pinned_bytes(root, "manifest.json", pin))
    if (manifest.get("run_type") != "training_run" or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] not in TRAINING_RUN_SCHEMAS
            or content_hash(manifest.get("config")) != content_hash(config)
            or manifest.get("model_id") != model or manifest.get("status") != "complete"
            or content_hash(manifest.get("seeds")) != content_hash(seeds)
            or manifest.get("seed_status") != {str(seed): "complete" for seed in seeds}
            or manifest.get("selection") != SELECTION_RULE
            or manifest.get("experiment_hashes", {}).get("manifest.json") != experiment_pin):
        raise ValueError(f"incompatible character checkpoint manifest for {model}")
    if is_bert(config):
        runtime = manifest.get("runtime", {})
        if any(runtime.get(name) != expected
               for name, expected in (item.split("==") for item in BERT_REQUIREMENTS)):
            raise ValueError(f"trained BERT runtime differs for {model}")
    for name in ("model.py", "bert.py") if is_bert(config) else ("model.py",):
        source = (Path(__file__).parents[1] / name).read_text(encoding="utf-8")
        source_pins = {hashlib.sha256(text.encode()).hexdigest()
                       for text in (source, source.replace("\n", "\r\n"))}
        if name == "model.py" and not model.startswith("syllable-acoustic-lambda-"):
            # The model before the syllable-acoustic head combination used the same existing heads.
            source_pins.add("4c9df43b451155582e1df117c0a5edaa60284a7b054fe34fad5cf75616ce6842")
        if not is_bert(config):
            # The pre-BERT model implements the same BiLSTM state and input contract.
            source_pins.update(("ee0a140f942238a9bdd99721ed1dcc6f4124bc4bb650cde33808cb49a7bcd12e",
                                "2807eb7ea7d05ebd160b66f24b8b54a70c497caf120257856e07eeaa93847311"))
        if manifest.get("code_sha256", {}).get(name) not in source_pins:
            raise ValueError(f"trained encoder source differs for {model}: {name}")
    artifacts = manifest["artifacts"]
    selections = {}
    for seed in seeds:
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


def expected_checkpoint_seeds(root, pin, model, experiment_pin, experiment):
    if model == "pretrained-bert":
        from .pretrained_source import selections
        return selections(root, pin, experiment_pin, experiment)
    manifest, selections = inspect_checkpoints(root, pin, model, experiment_pin, experiment)
    return {
        seed: {
            "checkpoint_sha256": manifest["artifacts"][f"seed-{seed}/best.pt"],
            "best_epoch": selected["best_epoch"],
            "seed_input_hashes": selected["seed_input_hashes"],
        }
        for seed, selected in selections.items()
    }


def expected_unknown_positions(root, pin, occurrences, test_pin):
    manifest = load_experiment_manifest(root, pin)
    seeds = manifest["identity"]["training_seeds"]
    artifacts = manifest["artifacts"]
    unknown = {}
    hashes = {}
    config = manifest["identity"]["config"]
    bert_unknown = None
    if is_bert(config):
        test_ids, samples = load_common_test(root, expected_manifest_sha256=pin)
        if content_hash(test_ids) != test_pin:
            raise ValueError("BERT evaluation test identity differs")
        rows = load_bert_inputs(config["architecture"]["encoder"]).prepare(samples, split="test")
        bert_unknown = np.asarray([rows[row["sample_id"]][2][row["character_position"]] for row in occurrences],
                                  dtype=bool)
    for seed in seeds:
        base = f"seeds/{seed}"
        if bert_unknown is None:
            name = f"{base}/vocabulary.json"
            vocabulary = json.loads(pinned_bytes(root, name, artifacts[name]))
            stoi, unk_id = vocabulary["stoi"], vocabulary["unk_id"]
            unknown[seed] = np.asarray([stoi.get(row["character"], unk_id) == unk_id for row in occurrences])
        else:
            unknown[seed] = bert_unknown
        name = f"{base}/input_hashes.json"
        hashes[seed] = json.loads(pinned_bytes(root, name, artifacts[name]))
        if hashes[seed]["test"] != test_pin:
            raise ValueError(f"experiment test identity differs for seed {seed}")
    return unknown, hashes


def extract_hidden_states(*, evaluation_root, evaluation_pin, experiment_root, experiment_pin,
                          training_root, training_pin, model, output, device, batch_size,
                          source_sha256, allow_execution=False):
    """Encode full test sentences, retaining only the sealed occurrence positions."""
    if not allow_execution:
        raise PermissionError("encoder extraction requires explicit execution authorization")
    if model == "pretrained-bert":
        from .pretrained_source import extract
        return extract(evaluation_root=evaluation_root, evaluation_pin=evaluation_pin,
            experiment_root=experiment_root, experiment_pin=experiment_pin, training_root=training_root,
            training_pin=training_pin, model=model, output=output, device=device, batch_size=batch_size,
            source_sha256=source_sha256, allow_execution=allow_execution)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    evaluation, rows = load_positions(evaluation_root, evaluation_pin)
    if evaluation["identity"]["experiment_manifest_sha256"] != experiment_pin:
        raise ValueError("evaluation and experiment pins differ")
    experiment = load_experiment_manifest(experiment_root, experiment_pin)
    training, selections = inspect_checkpoints(training_root, training_pin, model, experiment_pin, experiment)
    test_ids, samples = load_common_test(experiment_root, expected_manifest_sha256=experiment_pin)
    if content_hash(test_ids) != evaluation["identity"]["test_ids_sha256"]:
        raise ValueError("evaluation and experiment test identities differ")
    config = json.loads(pinned_bytes(experiment_root, "config.json", experiment["artifacts"]["config.json"]))
    labels = json.loads(pinned_bytes(experiment_root, "labels.json", experiment["artifacts"]["labels.json"]))
    if content_hash(config) != content_hash(training["config"]):
        raise ValueError("training and experiment configurations differ")
    bert_inputs = load_bert_inputs(config["architecture"]["encoder"]) if is_bert(config) else None
    bert_rows = bert_inputs.prepare(samples, split="test") if bert_inputs is not None else None
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
                "seeds": experiment["identity"]["training_seeds"], "source_sha256": source_sha256,
                "device": device, "batch_size": batch_size, "torch": torch.__version__}
    if bert_inputs is not None:
        identity["bert_runtime"] = {name: training["runtime"][name] for name in ("transformers", "tokenizers")}
    with staging_directory(target) as stage:
        for seed, selected in selections.items():
            base = f"seeds/{seed}"
            artifacts = experiment["artifacts"]
            vocab_bytes = pinned_bytes(experiment_root, f"{base}/vocabulary.json", artifacts[f"{base}/vocabulary.json"])
            vocabulary = json.loads(vocab_bytes)
            stoi, unk_id = vocabulary["stoi"], vocabulary["unk_id"]
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
            encoder = build_encoder(config, vocab, model_id=model, seed=seed,
                                    bert_inputs=bert_inputs, pretrained=False)
            encoder.load_state_dict(checkpoint["state_dict"], strict=True)
            encoder.to(device).eval()
            hidden = np.empty((len(rows), encoder.context_dim), dtype=np.float32)
            unknown = np.zeros(len(rows), dtype=bool)
            with torch.inference_mode():
                for start in range(0, len(samples), batch_size):
                    batch = samples[start:start+batch_size]
                    lengths = torch.tensor([len(sample["text"]) for sample in batch])
                    mask = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
                    if bert_inputs is not None:
                        aligned = [bert_rows[sample["sample_id"]] for sample in batch]
                        encoder_batch = bert_inputs.batch(aligned, mask)
                        unknown_rows = [row[2] for row in aligned]
                    else:
                        tokens = [[stoi.get(char, unk_id) for char in sample["text"]] for sample in batch]
                        ids = torch.zeros(mask.shape, dtype=torch.long)
                        for i, token_row in enumerate(tokens):
                            ids[i, :len(token_row)] = torch.tensor(token_row)
                        encoder_batch = {"token_ids": ids, "sentence_lengths": lengths, "char_mask": mask}
                        unknown_rows = [[token == unk_id for token in row] for row in tokens]
                    encoder_batch = {key: value.to(device) for key, value in encoder_batch.items()}
                    mask = encoder_batch["char_mask"]
                    values, result_mask = encoder.encode(encoder_batch)
                    if not torch.equal(mask, result_mask):
                        raise ValueError("encoder returned a different position mask")
                    values = values.detach().cpu().numpy()
                    for i, sample in enumerate(batch):
                        for index, position in by_sample[sample["sample_id"]]:
                            hidden[index] = values[i, position]
                            unknown[index] = unknown_rows[i][position]
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


def load_export(root, *, model, evaluation_pin, experiment_pin, occurrence_ids, training_pin,
                expected_seeds, expected_unknown=None, expected_input_hashes=None):
    if model == "pretrained-bert":
        from .pretrained_source import load
        yield from load(root, model=model, evaluation_pin=evaluation_pin, experiment_pin=experiment_pin,
            occurrence_ids=occurrence_ids, training_pin=training_pin, expected_seeds=expected_seeds,
            expected_unknown=expected_unknown, expected_input_hashes=expected_input_hashes)
        return
    seeds = list(expected_seeds)
    manifest = validate_bundle(root, EXTRACTION_TYPE)
    identity = manifest["identity"]
    if (identity.get("model") != model or identity.get("evaluation_manifest_sha256") != evaluation_pin
            or identity.get("experiment_manifest_sha256") != experiment_pin
            or content_hash(identity.get("seeds")) != content_hash(seeds)
            or identity.get("training_manifest_sha256") != training_pin):
        raise ValueError(f"hidden-state export identity mismatch for {model}")
    expected = {f"seed-{seed}/{name}" for seed in seeds
                for name in ("manifest.json", "occurrence_ids.json", "hidden.npy", "unknown.npy")}
    if set(manifest["artifacts"]) != expected:
        raise ValueError(f"hidden-state export artifact set differs for {model}")
    for seed in seeds:
        seed_root = Path(root) / f"seed-{seed}"
        child = validate_bundle(seed_root, SEED_TYPE)
        child_pin = manifest["artifacts"][f"seed-{seed}/manifest.json"]
        if json.loads(pinned_bytes(seed_root, "manifest.json", child_pin)) != child:
            raise ValueError(f"hidden-state seed manifest changed for {model}/{seed}")
        info = child["identity"]
        if info.get("seed") != seed or any(info.get(key) != value for key, value in identity.items()):
            raise ValueError(f"hidden-state seed identity mismatch for {model}/{seed}")
        if any(info.get(key) != expected_seeds[seed][key]
               for key in ("checkpoint_sha256", "best_epoch", "seed_input_hashes")):
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
        if ((expected_unknown is not None and not np.array_equal(unknown, expected_unknown[seed]))
                or (expected_input_hashes is not None and seed_hashes != expected_input_hashes[seed])):
            raise ValueError("encoder export differs from the pinned seed inputs")
        yield seed, hidden, unknown, info
