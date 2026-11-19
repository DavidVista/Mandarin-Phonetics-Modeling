"""A deterministic pretrained BERT source, pinned by the experiment snapshot."""
import io
import json
from pathlib import Path

import numpy as np

from ..artifacts import write_json
from ..bert import is_bert
from ..bert_snapshot import load_bert_inputs
from ..bundles import seal_bundle, staging_directory, validate_bundle
from ..experiment import load_common_test

MODEL = "pretrained-bert"
TYPE = "pretrained_encoder_export"


def selections(root, pin, experiment_pin, experiment):
    from .encoder_extraction import load_experiment_manifest, pinned_bytes
    if pin != experiment_pin or load_experiment_manifest(root, pin) != experiment:
        raise ValueError("pretrained source must be the pinned experiment snapshot")
    config = experiment["identity"]["config"]
    if not is_bert(config):
        raise ValueError("pretrained BERT requires a BERT experiment")
    snapshot = config["architecture"]["encoder"]
    return {seed: {"source_type": "pretrained_bert", "snapshot": snapshot,
        "checkpoint_sha256": snapshot["files"]["model.safetensors"], "best_epoch": None,
        "seed_input_hashes": json.loads(pinned_bytes(root, f"seeds/{seed}/input_hashes.json",
            experiment["artifacts"][f"seeds/{seed}/input_hashes.json"]))}
        for seed in experiment["identity"]["training_seeds"]}


def extract(*, evaluation_root, evaluation_pin, experiment_root, experiment_pin, training_root,
            training_pin, model, output, device, batch_size, source_sha256, allow_execution=False):
    if not allow_execution:
        raise PermissionError("pretrained extraction requires explicit execution authorization")
    if type(batch_size) is not int or batch_size < 1 or model != MODEL:
        raise ValueError("invalid pretrained extraction settings")
    from .encoder_extraction import load_positions, load_experiment_manifest
    experiment = load_experiment_manifest(experiment_root, experiment_pin)
    expected = selections(training_root, training_pin, experiment_pin, experiment)
    evaluation, rows = load_positions(evaluation_root, evaluation_pin)
    test_ids, samples = load_common_test(experiment_root, expected_manifest_sha256=experiment_pin)
    from ..bundles import content_hash
    if (evaluation["identity"]["experiment_manifest_sha256"] != experiment_pin or
            content_hash(test_ids) != evaluation["identity"]["test_ids_sha256"]):
        raise ValueError("pretrained evaluation inputs differ")
    target = Path(output).resolve()
    for root in (evaluation_root, experiment_root, training_root):
        path = Path(root).resolve()
        if target.is_relative_to(path) or path.is_relative_to(target):
            raise ValueError("pretrained output must be separate from inputs")
    inputs = load_bert_inputs(next(iter(expected.values()))["snapshot"], pretrained=True)
    aligned = inputs.prepare(samples, split="test")
    sample_map = {s["sample_id"]: s for s in samples}
    by_sample = {}
    for i, row in enumerate(rows):
        sample, pos = sample_map[row["sample_id"]], row["character_position"]
        if (sample["text"][pos] != row["character"] or sample["labels"][pos] != row["pinyin"]
                or row["occurrence_id"] != f"{row['sample_id']}:{pos}"):
            raise ValueError("pretrained occurrence alignment differs")
        by_sample.setdefault(row["sample_id"], []).append((i, pos))
    import torch
    from ..model import BertEncoder
    if torch.__version__.split("+", 1)[0] != "2.11.0":
        raise RuntimeError("encoder extraction requires Torch 2.11.0")
    encoder = BertEncoder(inputs, pretrained=True).to(device).eval()
    hidden = np.empty((len(rows), encoder.context_dim), dtype=np.float32)
    unknown = np.zeros(len(rows), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start+batch_size]
            mask = torch.arange(max(len(s["text"]) for s in batch))[None, :] < torch.tensor([len(s["text"]) for s in batch])[:, None]
            encoded = inputs.batch([aligned[s["sample_id"]] for s in batch], mask)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            values, result_mask = encoder.encode(encoded)
            if not torch.equal(result_mask, encoded["char_mask"]):
                raise ValueError("pretrained returned a different position mask")
            values = values.cpu().numpy()
            for j, sample in enumerate(batch):
                for i, pos in by_sample.get(sample["sample_id"], ()):
                    hidden[i] = values[j, pos]
                    unknown[i] = aligned[sample["sample_id"]][2][pos]
    if not np.isfinite(hidden).all() or np.any(np.linalg.norm(hidden, axis=1) == 0):
        raise ValueError("invalid pretrained hidden vectors")
    from importlib.metadata import version
    identity = dict(model=MODEL, evaluation_manifest_sha256=evaluation_pin,
        experiment_manifest_sha256=experiment_pin, source_manifest_sha256=training_pin,
        snapshot=next(iter(expected.values()))["snapshot"], source_sha256=source_sha256,
        representation="last layer pre-head", extraction_count=1, device=device, batch_size=batch_size,
        torch=torch.__version__, bert_runtime={name: version(name) for name in ("transformers", "tokenizers")})
    with staging_directory(target) as stage:
        write_json(stage / "occurrence_ids.json", [r["occurrence_id"] for r in rows])
        np.save(stage / "hidden.npy", hidden, allow_pickle=False)
        np.save(stage / "unknown.npy", unknown, allow_pickle=False)
        seal_bundle(stage, TYPE, identity)
    return target


def load(root, *, model, evaluation_pin, experiment_pin, occurrence_ids, training_pin,
         expected_seeds, expected_unknown=None, expected_input_hashes=None):
    from .encoder_extraction import pinned_bytes
    manifest = validate_bundle(root, TYPE)
    identity = manifest["identity"]
    if (model != MODEL or identity.get("model") != MODEL or
            identity.get("evaluation_manifest_sha256") != evaluation_pin or
            identity.get("experiment_manifest_sha256") != experiment_pin or
            identity.get("source_manifest_sha256") != training_pin or
            identity.get("extraction_count") != 1 or identity.get("representation") != "last layer pre-head"):
        raise ValueError("pretrained export provenance differs")
    artifacts = manifest["artifacts"]
    if set(artifacts) != {"occurrence_ids.json", "hidden.npy", "unknown.npy"}:
        raise ValueError("pretrained export artifacts differ")
    def read(name):
        return pinned_bytes(root, name, artifacts[name])
    ids = json.loads(read("occurrence_ids.json"))
    hidden = np.load(io.BytesIO(read("hidden.npy")), allow_pickle=False)
    unknown = np.load(io.BytesIO(read("unknown.npy")), allow_pickle=False)
    if (ids != occurrence_ids or hidden.shape != (len(ids), 768) or hidden.dtype != np.float32
            or not np.isfinite(hidden).all() or np.any(np.linalg.norm(hidden, axis=1) == 0)
            or unknown.shape != (len(ids),) or unknown.dtype != np.bool_):
        raise ValueError("pretrained export occurrence alignment differs")
    for seed, selected in expected_seeds.items():
        if (identity.get("snapshot") != selected.get("snapshot")
                or (expected_unknown is not None and not np.array_equal(unknown, expected_unknown[seed]))
                or (expected_input_hashes is not None and selected["seed_input_hashes"] != expected_input_hashes[seed])):
            raise ValueError("pretrained export source or inputs differ")
        yield seed, hidden, unknown, {**identity, "reused_for_seed": seed, "independent_training_seed": False}
