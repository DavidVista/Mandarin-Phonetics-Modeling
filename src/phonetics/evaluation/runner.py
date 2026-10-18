"""End-to-end orchestration for one immutable geometry-evaluation run."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_representations
from .bundle import load_evaluation_bundle
from .representations import random_occurrence_baseline, save_representations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_evaluation(
    bundle_path: str | Path, *, output_root: str | Path, run_id: str, adapter: str,
    code_provenance: dict[str, str], experiment_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None, device: str = "cpu", batch_size: int = 32,
    random_seed: int = 42, random_dimension: int | None = None,
) -> Path:
    """Extract or generate representations, then run all three analyses."""
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("run_id must be one safe path component")
    if adapter not in {"bilstm", "random"}:
        raise ValueError("adapter must be 'bilstm' or 'random'")
    if not isinstance(code_provenance, dict) or not code_provenance or any(
        not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for key, value in code_provenance.items()
    ):
        raise ValueError("explicit non-empty code_provenance hashes are required")
    bundle_root = Path(bundle_path).resolve()
    bundle = load_evaluation_bundle(bundle_root)
    target = (Path(output_root).resolve() / run_id).resolve()
    if target.parent != Path(output_root).resolve() or target.exists():
        raise FileExistsError(f"unsafe or existing evaluation run: {target}")
    if target == bundle_root or bundle_root in target.parents:
        raise ValueError("evaluation run output must not be inside its input bundle")
    if experiment_path is not None:
        experiment_root = Path(experiment_path).resolve()
        if target == experiment_root or experiment_root in target.parents:
            raise ValueError("evaluation run output must not be inside its source experiment")
    target.mkdir(parents=True)
    try:
        representations = target / "representations"
        if adapter == "bilstm":
            from .encoders import extract_bilstm

            if experiment_path is None or checkpoint_path is None:
                raise ValueError("bilstm adapter requires experiment_path and checkpoint_path")
            extract_bilstm(bundle_root, experiment_path=experiment_path, checkpoint_path=checkpoint_path,
                           output_dir=representations, device=device, batch_size=batch_size)
            adapter_config = {"experiment_path": str(Path(experiment_path).resolve()),
                              "checkpoint_path": str(Path(checkpoint_path).resolve()),
                              "device": device, "batch_size": batch_size}
        else:
            if random_dimension is None or random_dimension <= 0 or isinstance(random_dimension, bool):
                raise ValueError("random adapter requires a positive random_dimension")
            occurrence_ids = sorted(row["occurrence_id"] for row in bundle["occurrences"])
            hidden = random_occurrence_baseline(occurrence_ids, dimension=random_dimension, seed=random_seed)
            save_representations(
                representations, occurrence_ids=occurrence_ids, hidden=hidden, groups=bundle["groups"],
                provenance={"adapter": "iid_standard_normal", "seed": random_seed, "dimension": random_dimension,
                            "bundle_manifest_sha256": _sha256(bundle_root / "manifest.json")},
            )
            adapter_config = {"random_seed": random_seed, "random_dimension": random_dimension}
        analysis = analyze_representations(bundle_root, representations, output_dir=target / "analysis", run_label=run_id)
        actual = {
            "schema_version": 1, "run_id": run_id, "adapter": adapter, "adapter_config": adapter_config,
            "bundle_path": str(bundle_root), "bundle_manifest_sha256": _sha256(bundle_root / "manifest.json"),
            "representations_manifest_sha256": _sha256(representations / "manifest.json"),
            "analysis_manifest_sha256": _sha256(analysis / "manifest.json"), "code_provenance": code_provenance,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": _version("torch"), "dimsim": _version("dimsim"), "pypinyin": _version("pypinyin")},
        }
        (target / "actual_config.json").write_text(json.dumps(actual, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(target)
        raise
    return target
