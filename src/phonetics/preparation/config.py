"""Acoustic model configuration validation."""
from copy import deepcopy


def validate_model_config(model):
    _keys(model, {"id", "revision", "layer", "pooling", "sample_rate", "mono", "padding", "dtype", "device"}, {"processor"}, "model")
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
    return deepcopy(model)


def _keys(value, required, optional, name):
    if not required.issubset(value) or set(value) - required - optional:
        raise ValueError(f"invalid {name} keys")
