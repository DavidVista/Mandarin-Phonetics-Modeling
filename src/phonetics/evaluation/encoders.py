"""Strict adapters that align encoder outputs to model-independent occurrences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..model import build_encoder
from .bundle import load_evaluation_bundle
from .representations import save_representations, validate_matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_bilstm(
    bundle_path: str | Path, *, experiment_path: str | Path, checkpoint_path: str | Path,
    output_dir: str | Path, device: str = "cpu", batch_size: int = 32,
) -> Path:
    """Extract full-sentence BiLSTM outputs and save exact occurrence alignment."""
    if batch_size <= 0 or isinstance(batch_size, bool):
        raise ValueError("batch_size must be a positive integer")
    bundle = load_evaluation_bundle(bundle_path)
    experiment_root, checkpoint_file = Path(experiment_path).resolve(), Path(checkpoint_path).resolve()
    source_hashes = bundle["manifest"]["source_experiment"]["hashes"]
    for name in ("manifest.json", "config.json", "vocab.json", "samples.jsonl"):
        path = experiment_root / name
        if not path.is_file() or _sha256(path) != source_hashes.get(name):
            raise ValueError(f"experiment does not match evaluation source hash: {name}")
    config = json.loads((experiment_root / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((experiment_root / "vocab.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    expected = {
        "manifest_sha256": source_hashes["manifest.json"], "config_sha256": source_hashes["config.json"],
        "vocab_sha256": source_hashes["vocab.json"],
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint provenance does not match evaluation experiment")
    model = build_encoder(config, vocab)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device)).eval()

    sample_rows = [json.loads(line) for line in (experiment_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = {row["sample_id"]: row for row in sample_rows}
    occurrence_by_sample: dict[str, list[dict]] = {}
    for occurrence in bundle["occurrences"]:
        occurrence_by_sample.setdefault(occurrence["sample_id"], []).append(occurrence)
    sentence_ids = [row["sample_id"] for row in bundle["sentences"]]
    if set(sentence_ids) != set(occurrence_by_sample) or len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("bundle sentences and occurrence samples disagree")
    extracted: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(sentence_ids), batch_size):
            batch_ids = sentence_ids[start:start + batch_size]
            selected = [samples[sample_id] for sample_id in batch_ids]
            token_ids, offsets, cursor = [], [], 0
            for sample in selected:
                token_ids.extend(sample["token_ids"])
                offsets.extend(cursor + span[0] for span in sample["character_spans"])
                cursor += len(sample["token_ids"])
            lengths = torch.tensor([len(sample["character_spans"]) for sample in selected], dtype=torch.long)
            outputs = model(
                torch.tensor(token_ids, dtype=torch.long, device=device),
                torch.tensor(offsets, dtype=torch.long, device=device), lengths,
                return_hidden=True,
            )[-1].detach().cpu().numpy()
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
    hidden = np.stack([extracted[occurrence_id] for occurrence_id in occurrence_ids]).astype(np.float32)
    validate_matrix(occurrence_ids, hidden)
    provenance = {
        "adapter": "phonetics.evaluation.encoders.extract_bilstm", "device": str(device),
        "checkpoint_path": str(checkpoint_file), "checkpoint_sha256": _sha256(checkpoint_file),
        "bundle_manifest_sha256": _sha256(Path(bundle_path).resolve() / "manifest.json"),
        "experiment_hashes": expected, "hidden_dimension": int(hidden.shape[1]),
    }
    return save_representations(output_dir, occurrence_ids=occurrence_ids, hidden=hidden,
                                groups=bundle["groups"], provenance=provenance)
