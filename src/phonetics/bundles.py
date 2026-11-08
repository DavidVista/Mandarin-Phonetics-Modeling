"""Content-addressed validation and directory publication for pipeline stages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager

from .artifacts import safe_artifact, sha256_file, write_json


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_bundle(path, kind, *, identity=None):
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bundle must be a non-symlink directory")
    manifest_path = safe_artifact(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("bundle_type") != kind:
        raise ValueError("unsupported bundle schema or type")
    if manifest.get("identity_sha256") != content_hash(manifest.get("identity")):
        raise ValueError("bundle identity checksum mismatch")
    if identity is not None and manifest["identity"] != identity:
        raise ValueError("bundle inputs or parameters changed; use a new output ID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts or "manifest.json" in artifacts:
        raise ValueError("invalid bundle artifacts")
    actual = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("bundle must not contain symlinks")
        if item.is_file() and item != manifest_path:
            actual.add(item.relative_to(root).as_posix())
    if actual != set(artifacts):
        raise ValueError("bundle artifact set mismatch")
    for name, digest in artifacts.items():
        if sha256_file(safe_artifact(root, name)) != digest:
            raise ValueError(f"bundle artifact checksum mismatch: {name}")
    return manifest


def seal_bundle(root, kind, identity):
    root = Path(root)
    artifacts = {p.relative_to(root).as_posix(): sha256_file(p)
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(root / "manifest.json", {"schema_version": 1, "bundle_type": kind,
               "identity": identity, "identity_sha256": content_hash(identity), "artifacts": artifacts})
    validate_bundle(root, kind, identity=identity)


@contextmanager
def staging_directory(destination):
    """Stage on the destination filesystem; never replace existing bundles."""
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        yield temporary
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
