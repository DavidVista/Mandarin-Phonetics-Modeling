"""Build small, deterministic inputs for a local pipeline smoke run."""
from itertools import islice, product
from pathlib import Path

from .acoustic import assemble_character_data, validate_acoustic_targets
from .artifacts import write_json, write_jsonl
from .bundles import content_hash, seal_bundle, staging_directory
from .corpus import validate_character_corpus
from .data_contract import sample_id


def prepare_mock_data(config, paths):
    import numpy as np

    count = config["sample_size"] + config.get("test_size", config["sample_size"] // 10)
    dimension = config["architecture"]["acoustic_dim"]
    characters = {"\u5988": "ma1", "\u9a6c": "ma3", "\u591a": "duo1", "\u5b57": "zi4"}
    labels = sorted(characters.values())
    inventory = [{"label": label, "sha256": content_hash(label)} for label in labels]
    identity = {"inventory_manifest_sha256": content_hash(inventory), "backend": "synthetic"}
    corpus_path, acoustic_path = (Path(paths[key]) for key in ("corpus", "acoustic"))
    corpus_identity = {**identity, "sentence_count": count}
    acoustic_identity = {**identity, "dimension": dimension}
    if dimension < len(labels):
        raise ValueError("mock acoustic dimension must cover the synthetic labels")

    if not corpus_path.exists():
        width = 1
        while len(characters) ** width < count:
            width += 1
        rows = []
        for index, sentence in enumerate(islice(product(characters, repeat=width), count)):
            text = "".join(sentence)
            sequence = [characters[char] for char in text]
            rows.append({"text": text, "labels": sequence, "sample_id": sample_id(text, sequence),
                         "source_ids": [str(index)], "sentence_length": width, "positions": list(range(width))})
        with staging_directory(corpus_path) as stage:
            write_jsonl(stage / "corpus.jsonl", sorted(rows, key=lambda row: row["sample_id"]))
            write_jsonl(stage / "inventory.jsonl", inventory)
            write_jsonl(stage / "rejections.jsonl", [])
            write_json(stage / "statistics.json", {"input": count, "selected": count, "duplicates": 0, "rejected": 0})
            seal_bundle(stage, "character_corpus", corpus_identity)
            validate_character_corpus(stage, identity=corpus_identity)
    validate_character_corpus(corpus_path, identity=corpus_identity)

    if not acoustic_path.exists():
        with staging_directory(acoustic_path) as stage:
            write_jsonl(stage / "inventory.jsonl", inventory)
            write_json(stage / "labels.json", labels)
            np.save(stage / "targets.npy", np.eye(len(labels), dimension, dtype=np.float32), allow_pickle=False)
            write_json(stage / "inference.json", {"backend": "synthetic"})
            seal_bundle(stage, "acoustic_targets", acoustic_identity)
            validate_acoustic_targets(stage, identity=acoustic_identity)
    validate_acoustic_targets(acoustic_path, identity=acoustic_identity)
    return assemble_character_data(corpus_path, acoustic_path, paths["prepared"])
