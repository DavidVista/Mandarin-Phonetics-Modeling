"""Encoder and head configuration independent of the ML runtime."""
from __future__ import annotations

import math
BERT_MODEL_ID = "google-bert/bert-base-chinese"
BERT_REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
BERT_FILES = {
    "config.json": "88af762cfb15fa9ce183c84dcbfdc9d635196ba635dfe81f265fdd21d67f4c4c",
    "model.safetensors": "3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3",
    "tokenizer.json": "55c5edcfb8936895e019b2d8f97dcb547734f25b29f95009155f4c8c15649aae",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}
BERT_REQUIREMENTS = ("transformers==4.57.6", "tokenizers==0.22.2")


def validate_bert_snapshot(encoder):
    if (not isinstance(encoder, dict) or set(encoder) != {"type", "model_id", "revision", "files"}
            or encoder["type"] != "bert" or encoder["model_id"] != BERT_MODEL_ID
            or encoder["revision"] != BERT_REVISION):
        raise ValueError("BERT requires the Google Chinese checkpoint at the supported revision")
    if encoder["files"] != BERT_FILES:
        raise ValueError("BERT requires the verified SHA256 pins for the supported revision")


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
    if not isinstance(encoder, dict) or encoder.get("type") not in {"char_bilstm", "bert"}:
        raise ValueError("unsupported encoder type")
    if encoder["type"] == "bert":
        validate_bert_snapshot(encoder)
        dimensions = ()
    else:
        if set(encoder) != {"type", "embedding_dim", "hidden_per_direction", "layers"}:
            raise ValueError("invalid char_bilstm parameters")
        dimensions = (encoder["embedding_dim"], encoder["hidden_per_direction"], encoder["layers"])
    if not isinstance(heads, dict) or set(heads) != {"classification_hidden_dim", "classification_dropout"}:
        raise ValueError("invalid head parameters")
    for number in (*dimensions,
                   heads["classification_hidden_dim"], value["acoustic_dim"]):
        if type(number) is not int or number <= 0:
            raise ValueError("architecture dimensions and layer counts must be positive integers")
    dropout = heads["classification_dropout"]
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("classification_dropout must be finite and in [0, 1)")
    return {"encoder": dict(encoder), "heads": dict(heads), "acoustic_dim": value["acoustic_dim"]}
