"""Pinned public DIMSIM adapter used by evaluation-bundle preparation."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

DIMSIM_VERSION = "0.2.2"
PYPINYIN_VERSION = "0.55.0"


class DimsimDistance:
    """Validated callable around the public ``dimsim.get_distance`` API."""

    def __init__(self) -> None:
        versions = {
            "dimsim": metadata.version("dimsim"),
            "pypinyin": metadata.version("pypinyin"),
        }
        expected = {"dimsim": DIMSIM_VERSION, "pypinyin": PYPINYIN_VERSION}
        if versions != expected:
            raise RuntimeError(f"DIMSIM dependency versions {versions} do not match pinned {expected}")
        import dimsim

        if not callable(getattr(dimsim, "get_distance", None)):
            raise RuntimeError("dimsim.get_distance public API is unavailable")
        self._get_distance = dimsim.get_distance
        package_root = Path(dimsim.__file__).resolve().parent
        source_hashes = {}
        for path in sorted(package_root.rglob("*")):
            if path.is_file() and (path.suffix == ".py" or path.suffix.startswith(".pickle")):
                source_hashes[str(path.relative_to(package_root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.provenance = {
            "metric_name": "dimsim", "metric_version": DIMSIM_VERSION,
            "dependencies": versions, "api": "dimsim.get_distance([left], [right], pinyin=True)",
            "canonical_weights_modified": False,
            "package_file_sha256": source_hashes,
        }

    def __call__(self, left: str, right: str) -> float:
        # The public API expects utterances as lists even for single syllables.
        return float(self._get_distance([left], [right], pinyin=True))
