"""Character identity and acoustic target validation shared by the pipeline."""
import hashlib
import json


def sample_id(text, labels):
    return hashlib.sha256(json.dumps([text, labels], ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def is_han(character):
    code = ord(character)
    return (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF)


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
