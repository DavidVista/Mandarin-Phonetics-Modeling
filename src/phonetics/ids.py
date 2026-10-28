"""Pinned, deterministic IDS parsing and token mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import sha256_file

IDS_SCHEMA_VERSION = 1
IDS_RULES_VERSION = "ids-prefix-1"
PAD = "[PAD]"
UNKNOWN = "？"
OPERATORS = {chr(code) for code in range(0x2FF0, 0x2FFC)}
TERNARY_OPERATORS = {"⿲", "⿳"}
UNSUPPORTED_OPERATORS = {chr(code) for code in range(0x2FFC, 0x3000)} | {"㇯"}
EXPECTED_SNAPSHOT = "618dbab8a8ddefb958763c8b4afbaa741a4460de"
EXPECTED_LICENSE = "LGPL-3.0-or-later"
EXPECTED_ATTRIBUTION = "Make Me a Hanzi; dictionary.txt derived from Unihan and CJKlib"


def parse_expression(expression: str) -> tuple[str, ...]:
    if not isinstance(expression, str) or not expression:
        raise ValueError("IDS decomposition must be a non-empty string")
    if any(char in UNSUPPORTED_OPERATORS for char in expression):
        raise ValueError("unsupported IDS operator")
    cursor = 0
    def parse_node() -> None:
        nonlocal cursor
        if cursor >= len(expression):
            raise ValueError("IDS expression has too few operands")
        symbol = expression[cursor]
        cursor += 1
        if symbol in OPERATORS:
            arity = 3 if symbol in TERNARY_OPERATORS else 2
            for _ in range(arity):
                parse_node()
    parse_node()
    if cursor != len(expression):
        raise ValueError("IDS expression has trailing operands")
    return tuple(expression)


def read_dictionary(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid IDS JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"IDS line {line_number} must be an object")
            character, decomposition = row.get("character"), row.get("decomposition")
            if not isinstance(character, str) or len(character) != 1:
                raise ValueError(f"IDS line {line_number} character must be one code point")
            if character in mapping:
                raise ValueError(f"duplicate IDS character: {character}")
            if not isinstance(decomposition, str) or not decomposition:
                raise ValueError(f"empty or invalid IDS decomposition for {character}")
            parsed = parse_expression(decomposition)
            if UNKNOWN in decomposition:
                tokens, mode = (character,), "character_fallback"
            else:
                tokens, mode = parsed, "decomposition"
            mapping[character] = {"decomposition": decomposition, "tokens": list(tokens), "mode": mode}
    if not mapping:
        raise ValueError("IDS dictionary is empty")
    return dict(sorted(mapping.items(), key=lambda item: tuple(map(ord, item[0]))))


def validate_pin(spec: dict) -> tuple[Path, str]:
    expected = {"path", "sha256", "snapshot_commit", "license", "attribution"}
    if not isinstance(spec, dict) or set(spec) != expected:
        raise ValueError("IDS pin must contain path, sha256, snapshot_commit, license, attribution")
    path = Path(spec["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError("IDS dictionary checksum mismatch")
    if spec["snapshot_commit"] != EXPECTED_SNAPSHOT:
        raise ValueError("IDS snapshot commit mismatch")
    if spec["license"] != EXPECTED_LICENSE or not isinstance(spec["attribution"], str) or not spec["attribution"].strip():
        raise ValueError("IDS license or attribution mismatch")
    return path, digest


def build_mapping(spec: dict, *, dictionary_path: str | Path | None = None) -> tuple[dict, dict, dict]:
    if dictionary_path is None:
        path, digest = validate_pin(spec)
        records_path = path
    else:
        expected = {"path", "sha256", "snapshot_commit", "license", "attribution"}
        if not isinstance(spec, dict) or set(spec) != expected:
            raise ValueError("IDS pin must contain path, sha256, snapshot_commit, license, attribution")
        records_path = Path(dictionary_path).resolve()
        if not records_path.is_file() or sha256_file(records_path) != spec["sha256"]:
            raise ValueError("bundled IDS dictionary checksum mismatch")
        if spec["snapshot_commit"] != EXPECTED_SNAPSHOT or spec["license"] != EXPECTED_LICENSE or not str(spec["attribution"]).strip():
            raise ValueError("IDS source provenance mismatch")
        path, digest = Path(spec["path"]).resolve(), spec["sha256"]
    records = read_dictionary(records_path)
    tokens = sorted({token for record in records.values() for token in record["tokens"]}, key=ord)
    vocabulary = [PAD, *tokens]
    stoi = {token: index for index, token in enumerate(vocabulary)}
    encoded = {character: {**record, "ids": [stoi[token] for token in record["tokens"]]}
               for character, record in records.items()}
    provenance = {"path": str(path), "sha256": digest, "snapshot_commit": spec["snapshot_commit"],
                  "license": spec["license"], "attribution": spec["attribution"]}
    mapping_artifact = {"schema_version": IDS_SCHEMA_VERSION, "rules_version": IDS_RULES_VERSION,
                        "provenance": provenance, "characters": encoded}
    vocabulary_artifact = {"schema_version": IDS_SCHEMA_VERSION, "rules_version": IDS_RULES_VERSION,
                          "tokens": vocabulary, "stoi": stoi}
    return mapping_artifact, vocabulary_artifact, provenance


def encode_text(text: str, mapping: dict, vocabulary: dict) -> tuple[list[list[int]], list[int], str | None]:
    sequences, lengths = [], []
    for character in text:
        if character not in mapping["characters"]:
            raise ValueError(f"character is absent from IDS mapping: {character}")
        record = mapping["characters"][character]
        if len(record["ids"])!=len(record["tokens"]) or any(type(i) is not int or i<=0 or i>=len(vocabulary["tokens"]) or vocabulary["tokens"][i]!=token for token,i in zip(record["tokens"],record["ids"])):
            raise ValueError(f"invalid IDS token ID in mapping: {character}")
        sequences.append(list(record["ids"]))
        lengths.append(len(record["ids"]))
    return sequences, lengths, None


def content_hash(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

