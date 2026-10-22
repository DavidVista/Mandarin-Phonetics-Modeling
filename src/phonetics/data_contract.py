"""Shared prepared-data eligibility rules, independent of experiment splitting."""

from __future__ import annotations

import hashlib
import json

from .pinyin import PINYIN_CONTRACT_VERSION, canonical_labels, decompose

DATA_CONTRACT_VERSION = "prepared-1"
FILTERS = ("nonempty", "characters", "alignment", "pinyin", "wubi", "acoustic")


def sample_id(text, labels):
    return hashlib.sha256(json.dumps([text, labels], ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def is_han(character):
    code = ord(character)
    return (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF)


def pair_reason(text, labels, wubi, acoustic, *, stage=None):
    """Return the first failing stage; a supplied stage checks only that rule."""
    stages = FILTERS if stage is None else (stage,)
    for current in stages:
        if current == "nonempty":
            if not isinstance(text, str) or not text or not labels:
                return current
        elif current == "characters":
            if any(not is_han(ch) for ch in text):
                return current
        elif current == "alignment":
            if len(text) != len(labels):
                return current
        elif current == "pinyin":
            try:
                for label in labels:
                    decompose(label)
            except (ValueError, TypeError, IndexError):
                return current
        elif current == "wubi":
            if any(ch not in wubi for ch in text):
                return current
        elif current == "acoustic":
            if any(label not in acoustic for label in labels):
                return current
        else:
            raise ValueError(f"unknown filter stage: {current}")
    return None


def validate_wubi(mapping, *, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"):
    """Validate supplied codes without stripping index digits or reversing maps."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Wubi mapping must be a non-empty dict")
    for key, value in mapping.items():
        if (not isinstance(key, str) or len(key) != 1 or not isinstance(value, str)
                or not value or any(symbol not in alphabet for symbol in value)):
            raise ValueError(f"invalid Wubi entry: {key!r}")
    return dict(sorted(mapping.items()))


def validate_acoustic(mapping, *, expected_keys=None):
    import numpy as np

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("acoustic mapping must be a non-empty dict")
    if any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError("invalid acoustic key")
    if expected_keys is not None and set(mapping) != set(expected_keys):
        raise ValueError("acoustic keys disagree with inventory")
    dimension = None
    for key, value in sorted(mapping.items()):
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or not vector.size:
            raise ValueError(f"invalid acoustic vector: {key}")
        if dimension is None:
            dimension = vector.size
        if (vector.size != dimension or not np.isfinite(vector).all()
                or np.linalg.norm(vector.astype(np.float64)) <= 1e-12):
            raise ValueError(f"invalid acoustic vector: {key}")
    return dimension
