"""Strict local readers. Inputs are never silently repaired."""

from __future__ import annotations

import csv
import pickle
import wave
from pathlib import Path

from ..artifacts import sha256_file
from ..data_contract import validate_wubi
from ..pinyin import canonical_labels


def verify_source(path, expected=None):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path) if path.is_file() else None
    if expected is not None and digest != expected:
        raise ValueError(f"source checksum mismatch: {path}")
    return path, digest


def read_corpus(spec):
    path, digest = verify_source(spec["path"], spec.get("sha256"))
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        allowed = {"text", "pinyin", "source_id"}
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"text", "pinyin"}.issubset(reader.fieldnames) or set(reader.fieldnames) - allowed):
            raise ValueError("corpus headers must be text,pinyin[,source_id]")
        for number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"malformed corpus row {number}")
            raw = row["pinyin"].split() if isinstance(row["pinyin"], str) else []
            prefix = spec.get("source_id")
            local_id = row.get("source_id") or f"row-{number}"
            source_id = f"{prefix}:{local_id}" if prefix else local_id
            rows.append({"text": row["text"] if isinstance(row["text"], str) else "",
                         "raw_pinyin": raw,
                         "source_id": source_id, "source_row": number})
    return rows, {"path": str(path), "sha256": digest}


def read_wubi(spec):
    path, digest = verify_source(spec["path"], spec.get("sha256"))
    with path.open("rb") as handle:
        mapping = pickle.load(handle)
    return validate_wubi(mapping, alphabet=spec.get("alphabet", "abcdefghijklmnopqrstuvwxyz0123456789")), {"path": str(path), "sha256": digest}


def read_sounds(spec):
    root = Path(spec["path"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = spec.get("manifest")
    entries = []
    if manifest_path:
        manifest, manifest_hash = verify_source(manifest_path, spec.get("sha256"))
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != ["path", "pinyin"]:
                raise ValueError("sound manifest headers must be path,pinyin")
            entries = [(row["path"], row["pinyin"]) for row in reader]
        source_hash = manifest_hash
    else:
        files = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}), key=lambda p: p.name)
        entries = [(p.name, p.stem) for p in files]
        source_hash = None
        if spec.get("sha256") is not None:
            raise ValueError("sounds sha256 requires a manifest file")
    inventory, seen = [], {}
    for relative, raw_label in entries:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path.suffix.lower() not in {".wav", ".mp3"}:
            raise ValueError(f"unsafe or unsupported sound path: {relative}")
        label = canonical_labels([raw_label], tone_convention=spec["tone_convention"])[0]
        if label in seen:
            raise ValueError(f"canonical sound collision: {label}: {seen[label]} and {relative}")
        seen[label] = relative
        info = audio_info(path)
        inventory.append({"label": label, "raw_label": raw_label, "path": str(path), "relative_path": relative,
                          "sha256": sha256_file(path), **info})
    if not inventory:
        raise ValueError("sound inventory is empty")
    return sorted(inventory, key=lambda row: (row["label"], row["relative_path"])), {"path": str(root), "sha256": source_hash}


def audio_info(path):
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            rate, channels, frames, width = handle.getframerate(), handle.getnchannels(), handle.getnframes(), handle.getsampwidth()
            raw = handle.readframes(frames)
            if rate <= 0 or channels <= 0 or frames <= 0 or len(raw) != frames * channels * width:
                raise ValueError(f"invalid or truncated WAV: {path}")
            return {"sample_rate": rate, "channels": channels, "frames": frames, "duration_seconds": frames / rate}
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {"sample_rate": info.samplerate, "channels": info.channels,
                "frames": info.frames, "duration_seconds": info.duration}
    except ImportError as exc:
        raise RuntimeError("soundfile is required to inspect MP3 inputs") from exc
