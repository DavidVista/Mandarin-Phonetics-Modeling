"""Small, deterministic artifact helpers for prepared datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                                  + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object")
        result.append(value)
    return result


def safe_artifact(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe artifact path: {relative}")
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"symlink artifact path: {relative}")
    return path
