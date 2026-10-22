"""Strict configuration parsing for prepared datasets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

SCHEMA_VERSION = 1


def validate_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("config must be a dict")
    required = {"dataset_id", "output_root", "cache_root", "corpus", "sounds", "wubi", "model", "sampling"}
    if set(value) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    config = deepcopy(value)
    dataset_id = config["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id or Path(dataset_id).name != dataset_id or dataset_id in {".", ".."}:
        raise ValueError("dataset_id must be one safe path component")
    for section in ("corpus", "sounds", "wubi", "model", "sampling"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be an object")
    _keys(config["corpus"], {"path", "tone_convention"}, {"sha256", "source_id"}, "corpus")
    _keys(config["sounds"], {"path", "tone_convention"}, {"sha256", "manifest"}, "sounds")
    _keys(config["wubi"], {"path"}, {"sha256", "alphabet"}, "wubi")
    _keys(config["model"], {"id", "revision", "layer", "pooling", "sample_rate", "mono", "padding", "dtype", "device"}, {"processor"}, "model")
    _keys(config["sampling"], {"size", "strategy", "seed"}, set(), "sampling")
    if config["corpus"]["tone_convention"] not in {"explicit", "neutral_unmarked"} or config["sounds"]["tone_convention"] not in {"explicit", "neutral_unmarked"}:
        raise ValueError("tone_convention is required and must be explicit or neutral_unmarked")
    model = config["model"]
    if not isinstance(model["revision"], str) or not model["revision"]:
        raise ValueError("model revision is required")
    if model["pooling"] not in {"mean_all", "mean_valid"} or model["padding"] not in {"legacy_one_second", "none"}:
        raise ValueError("unsupported pooling or padding")
    if model["pooling"] == "mean_valid":
        raise ValueError("mean_valid is unsupported until padding masks can be proven correct")
    if model["dtype"] != "float32" or not isinstance(model["mono"], bool) or not isinstance(model["layer"], int) or isinstance(model["layer"], bool):
        raise ValueError("model requires dtype=float32, boolean mono, and integer layer")
    if not isinstance(model["device"], str) or not model["device"]:
        raise ValueError("model device must be explicit")
    if not isinstance(model["sample_rate"], int) or isinstance(model["sample_rate"], bool) or model["sample_rate"] <= 0:
        raise ValueError("sample_rate must be a positive integer")
    sampling = config["sampling"]
    if sampling["strategy"] != "uniform" or not isinstance(sampling["seed"], int) or isinstance(sampling["seed"], bool):
        raise ValueError("sampling requires strategy=uniform and an integer seed")
    size = sampling["size"]
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        raise ValueError("sampling size must be null or a positive integer")
    return config


def _keys(value, required, optional, name):
    if not required.issubset(value) or set(value) - required - optional:
        raise ValueError(f"invalid {name} keys")
