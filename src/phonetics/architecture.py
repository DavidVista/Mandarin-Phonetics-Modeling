"""Encoder and head configuration independent of the ML runtime."""
from __future__ import annotations

import math


def default_architecture(acoustic_dim=1024):
    return {"encoder": {"type": "char_bilstm", "embedding_dim": 64,
                        "hidden_per_direction": 64, "layers": 1},
            "heads": {"classification_hidden_dim": 64, "classification_dropout": 0.5},
            "acoustic_dim": acoustic_dim}


def normalize_architecture(value):
    """Read existing flat bundle configs as well as explicit encoder configs."""
    if not isinstance(value, dict):
        raise ValueError("architecture must be an object")
    if "encoder" not in value:
        keys = {"embedding_dim", "sentence_bilstm_hidden_per_direction", "sentence_bilstm_layers",
                "context_dim", "classification_hidden_dim", "classification_dropout", "acoustic_dim"}
        if set(value) != keys or value["context_dim"] != 2 * value["sentence_bilstm_hidden_per_direction"]:
            raise ValueError("invalid flat architecture contract")
        value = {"encoder": {"type": "char_bilstm", "embedding_dim": value["embedding_dim"],
                            "hidden_per_direction": value["sentence_bilstm_hidden_per_direction"],
                            "layers": value["sentence_bilstm_layers"]},
                 "heads": {k: value[k] for k in ("classification_hidden_dim", "classification_dropout")},
                 "acoustic_dim": value["acoustic_dim"]}
    if set(value) != {"encoder", "heads", "acoustic_dim"}:
        raise ValueError("architecture requires encoder, heads and acoustic_dim")
    encoder, heads = value["encoder"], value["heads"]
    if not isinstance(encoder, dict) or encoder.get("type") != "char_bilstm":
        raise ValueError("unsupported encoder type; only char_bilstm is implemented")
    if set(encoder) != {"type", "embedding_dim", "hidden_per_direction", "layers"}:
        raise ValueError("invalid char_bilstm parameters")
    if not isinstance(heads, dict) or set(heads) != {"classification_hidden_dim", "classification_dropout"}:
        raise ValueError("invalid head parameters")
    for number in (encoder["embedding_dim"], encoder["hidden_per_direction"], encoder["layers"],
                   heads["classification_hidden_dim"], value["acoustic_dim"]):
        if type(number) is not int or number <= 0:
            raise ValueError("architecture dimensions and layer counts must be positive integers")
    dropout = heads["classification_dropout"]
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("classification_dropout must be finite and in [0, 1)")
    return {"encoder": dict(encoder), "heads": dict(heads), "acoustic_dim": value["acoustic_dim"]}
