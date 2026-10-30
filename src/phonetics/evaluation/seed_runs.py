"""Explicit, integrity-checked discovery of seeds in a training run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..objectives import canonical_model_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_seed_runs(run_dir: str | Path, expected_model_id: str,
                       expected_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    """Return sorted seed/checkpoint records from exactly the selected run directory."""
    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("training run path must be a directory")
    model_id = canonical_model_id(expected_model_id)
    if not isinstance(expected_seeds, (tuple, list)) or not expected_seeds:
        raise ValueError("expected_seeds must be a non-empty sequence of integers")
    if any(type(seed) is not int for seed in expected_seeds) or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected_seeds must contain distinct integers")
    expected_seeds = tuple(sorted(expected_seeds))
    if len(expected_seeds) != 5 or any(seed < 0 for seed in expected_seeds):
        raise ValueError("expected_seeds must contain exactly five non-negative seed IDs")
    manifest_path = root / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"training run manifest is missing or unreadable: {manifest_path}") from error
    manifest = json.loads(raw)
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("run_type") != "training_run"
            or manifest.get("status") != "complete" or manifest.get("run_id") != root.name
            or manifest.get("model_id") != model_id):
        raise ValueError("training run manifest identity or completion status is invalid")
    seeds = manifest.get("seeds")
    if (not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds)
            or len(seeds) != len(set(seeds)) or tuple(sorted(seeds)) != expected_seeds):
        raise ValueError("training run seed inventory differs from expected seeds")
    statuses = manifest.get("seed_status")
    if not isinstance(statuses, dict) or statuses != {str(seed): "complete" for seed in seeds}:
        raise ValueError("training run does not have complete status for every seed")
    seed_dirs = {path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("seed-")}
    if seed_dirs != {f"seed-{seed}" for seed in expected_seeds}:
        raise ValueError("training run seed directory inventory is not exact")
    artifact_hashes = manifest.get("artifacts")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("training run artifact inventory is missing")
    records = []
    for seed in expected_seeds:
        relative = Path(f"seed-{seed}") / "best.pt"
        try:
            checkpoint = (root / relative).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"missing checkpoint for model {model_id}, seed {seed}: {root / relative}") from error
        if root not in checkpoint.parents or not checkpoint.is_file():
            raise ValueError("checkpoint path escapes training run or is not a file")
        expected_hash = artifact_hashes.get(relative.as_posix())
        actual_hash = _sha256(checkpoint)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise ValueError(f"training checkpoint checksum mismatch: seed {seed}")
        records.append({"seed": seed, "checkpoint_path": str(checkpoint), "checkpoint_sha256": actual_hash})
    experiment_hashes = manifest.get("experiment_hashes")
    required = {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"}
    if (not isinstance(experiment_hashes, dict) or set(experiment_hashes) != required
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in experiment_hashes.values())):
        raise ValueError("training source experiment hash inventory is invalid")
    return {"run_dir": str(root), "run_id": manifest["run_id"], "model_id": model_id,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "experiment_hashes": experiment_hashes,
            "seeds": records}
