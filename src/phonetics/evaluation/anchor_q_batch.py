"""Batch scoring for explicitly selected complete training runs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..objectives import canonical_model_id
from .anchor_q import score_anchor_q
from .bundle import load_evaluation_bundle
from .representations import load_contextual_representations
from .seed_runs import discover_seed_runs


def score_anchor_q_batch(bundle_path: str | Path, run_dirs: dict[str, str | Path],
                         representation_dirs: dict[str, dict[int, str | Path]]) -> dict[str, Any]:
    """Score all five explicit seeds per model against one verified schema-3 bundle."""
    if not isinstance(run_dirs, dict) or not run_dirs or not isinstance(representation_dirs, dict):
        raise ValueError("explicit run and representation mappings are required")
    canonical_runs = {canonical_model_id(key): value for key, value in run_dirs.items()}
    canonical_reps = {canonical_model_id(key): value for key, value in representation_dirs.items()}
    if (len(canonical_runs) != len(run_dirs) or len(canonical_reps) != len(representation_dirs)
            or set(canonical_runs) != set(canonical_reps)):
        raise ValueError("run and representation model inventories must match uniquely")
    bundle = load_evaluation_bundle(bundle_path)
    if bundle["manifest"].get("schema_version") != 3:
        raise ValueError("batch anchor-Q requires a schema-3 bundle")
    bundle_root = Path(bundle_path).resolve()
    bundle_sha = hashlib.sha256((bundle_root / "manifest.json").read_bytes()).hexdigest()
    source_hashes = bundle["manifest"].get("source_experiment", {}).get("hashes")
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != {"manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"}):
        raise ValueError("schema-3 bundle source experiment hashes are missing")
    scored = []
    for model_id in sorted(canonical_runs):
        try:
            run = discover_seed_runs(canonical_runs[model_id], model_id)
        except (OSError, ValueError) as error:
            raise ValueError(f"training run validation failed for model {model_id}: {error}") from error
        rep_seeds = canonical_reps[model_id]
        if (not isinstance(rep_seeds, dict) or any(type(seed) is not int for seed in rep_seeds)
                or set(rep_seeds) != {0, 1, 2, 3, 4}):
            raise ValueError(f"representations must explicitly provide seeds 0 through 4 for {model_id}")
        for record in run["seeds"]:
            seed = record["seed"]
            try:
                loaded = load_contextual_representations(rep_seeds[seed])
            except (OSError, ValueError) as error:
                raise ValueError(f"representation validation failed for model {model_id}, seed {seed}, path {rep_seeds[seed]}: {error}") from error
            provenance = loaded["manifest"]["provenance"]
            if provenance["model_id"] != model_id or provenance["seed"] != seed:
                raise ValueError(f"representation model/seed mismatch: {model_id} seed {seed}")
            if provenance["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise ValueError(f"representation checkpoint mismatch: {model_id} seed {seed}")
            if provenance["experiment_hashes"] != run["experiment_hashes"]:
                raise ValueError(f"representation training source hashes mismatch: {model_id} seed {seed}")
            if any(run["experiment_hashes"].get(name) != digest for name, digest in source_hashes.items()):
                raise ValueError("evaluation bundle and training run source experiment hashes differ")
            result = score_anchor_q(bundle, loaded["occurrence_ids"], loaded["hidden"], provenance,
                                    bundle_manifest_sha256=bundle_sha,
                                    tie_tolerance=bundle["config"]["scoring"]["anchor_q"]["tie_tolerance"])
            scored.append({"model_id": model_id, "seed": seed, "run_id": run["run_id"],
                           "run_manifest_sha256": run["manifest_sha256"],
                           "checkpoint_path": record["checkpoint_path"],
                           "checkpoint_sha256": record["checkpoint_sha256"],
                           "experiment_hashes": run["experiment_hashes"], "score": result})
    return {"schema_version": 1, "bundle_manifest_sha256": bundle_sha, "results": scored}
