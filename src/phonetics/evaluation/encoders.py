"""Strict adapters that align encoder outputs to model-independent occurrences."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from ..data import collate_experiment
from ..experiment import load_experiment
from ..model import build_encoder
from ..objectives import canonical_model_id, parse_model_id
from .bundle import load_evaluation_bundle
from .representations import save_contextual_representations, save_representations, validate_matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_hash_contract(source_hashes: dict, experiment_manifest: dict) -> dict:
    """Complete evaluation's five-file source pin with its validated acoustic artifact pin."""
    required = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"}
    if not isinstance(source_hashes, dict) or set(source_hashes) != required:
        raise ValueError("evaluation source experiment hash set is invalid")
    acoustic_hash = experiment_manifest.get("artifacts", {}).get("acoustic_targets.npy")
    if not isinstance(acoustic_hash, str) or len(acoustic_hash) != 64 or any(c not in "0123456789abcdef" for c in acoustic_hash):
        raise ValueError("validated experiment manifest lacks acoustic target hash")
    return {**source_hashes, "acoustic_targets.npy": acoustic_hash}


def extract_bilstm(
    bundle_path: str | Path, *, experiment_path: str | Path, checkpoint_path: str | Path,
    output_dir: str | Path, device: str = "cpu", batch_size: int = 32, contextual_only: bool = False,
    expected_model_id: str | None = None, expected_seed: int | None = None,
    source_sha256: str | None = None,
) -> Path:
    """Extract full-sentence BiLSTM outputs and save exact occurrence alignment."""
    if not isinstance(batch_size, int) or batch_size <= 0 or isinstance(batch_size, bool):
        raise ValueError("batch_size must be a positive integer")
    if source_sha256 is not None and (not isinstance(source_sha256, str) or len(source_sha256) != 64
                                      or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    bundle = load_evaluation_bundle(bundle_path)
    experiment_root, checkpoint_file = Path(experiment_path).resolve(), Path(checkpoint_path).resolve()
    source_hashes = bundle["manifest"]["source_experiment"]["hashes"]
    for name in ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"):
        path = experiment_root / name
        if not path.is_file() or _sha256(path) != source_hashes.get(name):
            raise ValueError(f"experiment does not match evaluation source hash: {name}")
    experiment = load_experiment(experiment_root)
    expected_checkpoint_hashes = _checkpoint_hash_contract(source_hashes, experiment["manifest"])
    config, vocab = experiment["config"], experiment["vocab"]
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    model_id = canonical_model_id(checkpoint.get("model_id"))
    seed = checkpoint.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("checkpoint seed must be an integer")
    if expected_model_id is not None and canonical_model_id(expected_model_id) != model_id:
        raise ValueError("checkpoint model_id does not match expected_model_id")
    if expected_seed is not None and (not isinstance(expected_seed, int) or isinstance(expected_seed, bool) or expected_seed != seed):
        raise ValueError("checkpoint seed does not match expected_seed")
    _, lambda_cosine = parse_model_id(model_id)
    if "lambda_cosine" in checkpoint and checkpoint["lambda_cosine"] != lambda_cosine:
        raise ValueError("checkpoint lambda disagrees with model_id")
    if checkpoint.get("experiment_hashes") != expected_checkpoint_hashes:
        raise ValueError("checkpoint provenance does not match evaluation experiment")
    if checkpoint.get("config_sha256") != source_hashes["config.json"] or checkpoint.get("vocab_sha256") != source_hashes["vocab.json"]:
        raise ValueError("checkpoint config or vocabulary provenance mismatch")
    model = build_encoder(config, vocab, model_id=model_id, seed=seed)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device)).eval()

    samples = experiment["samples"]
    occurrence_by_sample: dict[str, list[dict]] = {}
    for occurrence in bundle["occurrences"]:
        occurrence_by_sample.setdefault(occurrence["sample_id"], []).append(occurrence)
    sentence_key = "sentence_id" if bundle["manifest"].get("schema_version") == 3 else "sample_id"
    sentence_ids = [row[sentence_key] for row in bundle["sentences"]]
    if set(sentence_ids) != set(occurrence_by_sample) or len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("bundle sentences and occurrence samples disagree")
    extracted: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(sentence_ids), batch_size):
            batch_ids = sentence_ids[start:start + batch_size]
            selected = [samples[sample_id] for sample_id in batch_ids]
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in collate_experiment(selected).items()}
            outputs, output_mask = model.encode(batch)
            expected_mask = batch["char_mask"]
            context_dimension = getattr(model, "context_dim", None)
            if (not torch.is_tensor(outputs) or not torch.is_tensor(output_mask)
                    or not torch.is_tensor(expected_mask) or expected_mask.dtype != torch.bool
                    or outputs.ndim != 3 or outputs.shape[:2] != expected_mask.shape
                    or not isinstance(context_dimension, int) or outputs.shape[2] != context_dimension
                    or output_mask.shape != expected_mask.shape or output_mask.dtype != torch.bool):
                raise ValueError("encoder output shape or mask shape/type is invalid")
            outputs = outputs.detach().cpu().numpy()
            if not torch.equal(output_mask, batch["char_mask"]):
                raise ValueError("encoder output mask differs from source positions")
            for row_index, (sample_id, sample) in enumerate(zip(batch_ids, selected)):
                expected_positions = sorted(occurrence_by_sample[sample_id], key=lambda row: row["character_position"])
                if [row["character_position"] for row in expected_positions] != list(range(len(sample["text"]))) or any(
                    row["character"] != sample["text"][row["character_position"]] for row in expected_positions
                ):
                    raise ValueError(f"bundle occurrences do not exactly cover source sentence {sample_id}")
                for occurrence in expected_positions:
                    extracted[occurrence["occurrence_id"]] = outputs[row_index, occurrence["character_position"]]
    occurrence_ids = sorted(row["occurrence_id"] for row in bundle["occurrences"])
    if set(extracted) != set(occurrence_ids):
        raise ValueError("encoder extraction did not produce every expected occurrence")
    hidden = np.stack([extracted[occurrence_id] for occurrence_id in occurrence_ids])
    validate_matrix(occurrence_ids, hidden)
    provenance = {
        "adapter": "phonetics.evaluation.encoders.extract_bilstm", "device": str(device),
        "checkpoint_path": str(checkpoint_file), "checkpoint_sha256": _sha256(checkpoint_file),
        "bundle_manifest_sha256": _sha256(Path(bundle_path).resolve() / "manifest.json"),
        "experiment_hashes": expected_checkpoint_hashes, "config_sha256": source_hashes["config.json"],
        "vocab_sha256": source_hashes["vocab.json"], "model_id": model_id,
        "seed": seed, "hidden_dimension": int(hidden.shape[1]),
        "evaluation_mode": model.training is False, "output_dtype": str(hidden.dtype),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "device": str(device)},
        "source_sha256": source_sha256 or _sha256(Path(__file__).resolve()),
    }
    if contextual_only:
        return save_contextual_representations(output_dir, occurrence_ids=occurrence_ids,
                                               hidden=hidden, provenance=provenance)
    return save_representations(output_dir, occurrence_ids=occurrence_ids, hidden=hidden,
                                groups=bundle["groups"], provenance=provenance)
