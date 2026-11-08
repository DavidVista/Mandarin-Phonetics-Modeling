"""Prepare the complete eligible character corpus without IDS or embeddings."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .artifacts import read_jsonl, sha256_file, write_json, write_jsonl
from .bundles import seal_bundle, staging_directory, validate_bundle
from .data_contract import is_han, sample_id
from .pinyin import PARSER_PROVENANCE, canonical_labels, validate_canonical_label
from .sound_inventory import validate_sound_inventory


def prepare_character_corpus(csv_path, inventory_path, destination, *, expected_sha256,
                             tone_convention="explicit", source_id="duyu"):
    csv_path, inventory_path = Path(csv_path), Path(inventory_path)
    if tone_convention not in {"explicit", "neutral_unmarked"}:
        raise ValueError("unsupported tone convention")
    if sha256_file(csv_path) != expected_sha256:
        raise ValueError("corpus source checksum mismatch")
    validate_sound_inventory(inventory_path)
    inventory = {r["label"] for r in read_jsonl(inventory_path / "inventory.jsonl")}
    identity = {"corpus_sha256": expected_sha256, "source_id": source_id,
                "inventory_manifest_sha256": sha256_file(inventory_path / "manifest.json"),
                "parser": PARSER_PROVENANCE, "tone_convention": tone_convention,
                "conflict_policy": "exclude_all_variants", "contract": "character-corpus-1"}
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        validate_character_corpus(target, identity=identity)
        return target
    accepted, rejected, source_ids = [], [], set()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames
        if (not fields or len(set(fields)) != len(fields) or not {"text", "pinyin"}.issubset(fields)
                or set(fields) - {"text", "pinyin", "source_id"}):
            raise ValueError("corpus headers must be text,pinyin[,source_id]")
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"malformed corpus row {number}")
            sid = f"{source_id}:{raw.get('source_id') or f'row-{number}'}"
            if sid in source_ids:
                raise ValueError(f"duplicate corpus source ID: {sid}")
            source_ids.add(sid)
            text, tokens = raw["text"], raw["pinyin"].split()
            row = {"text": text, "source_ids": [sid], "source_row": number}
            reason = None
            if not text or not tokens:
                reason = "nonempty"
            elif any(not is_han(c) for c in text):
                reason = "characters"
            elif len(text) != len(tokens):
                reason = "alignment"
            try:
                labels = canonical_labels(tokens, tone_convention=tone_convention)
            except (ValueError, TypeError):
                labels = tokens
                reason = reason or "pinyin"
            row["labels"] = labels
            if reason:
                rejected.append({**row, "reason": reason})
            else:
                accepted.append(row)
    # Pronunciation conflicts are excluded before sound-coverage filtering, so
    # losing one reading to coverage cannot silently resolve an ambiguity.
    groups = defaultdict(list)
    for row in accepted:
        groups[row["text"]].append(row)
    rows = []
    duplicate_count = 0
    for text, members in sorted(groups.items()):
        variants = {tuple(row["labels"]) for row in members}
        if len(variants) > 1:
            rejected.extend({**row, "reason": "conflicting_pronunciation"} for row in members)
            continue
        labels = members[0]["labels"]
        if not set(labels).issubset(inventory):
            rejected.extend({**row, "reason": "sound_coverage",
                             "missing_labels": sorted(set(labels) - inventory)} for row in members)
            continue
        duplicate_count += len(members) - 1
        rows.append({"text": text, "labels": labels,
                     "source_ids": sorted(s for row in members for s in row["source_ids"]),
                     "sample_id": sample_id(text, labels), "sentence_length": len(text),
                     "positions": list(range(len(text)))})
    rows.sort(key=lambda row: row["sample_id"])
    if not rows:
        raise ValueError("no eligible corpus sentences remain")
    with staging_directory(target) as temporary:
        write_jsonl(temporary / "corpus.jsonl", rows)
        write_jsonl(temporary / "rejections.jsonl", rejected)
        write_jsonl(temporary / "inventory.jsonl", read_jsonl(inventory_path / "inventory.jsonl"))
        write_json(temporary / "statistics.json", {"input": len(source_ids), "selected": len(rows),
                   "rejected": len(rejected), "duplicates": duplicate_count,
                   "reasons": dict(sorted(Counter(r["reason"] for r in rejected).items()))})
        seal_bundle(temporary, "character_corpus", identity)
        validate_character_corpus(temporary)
    return target


def validate_character_corpus(path, *, identity=None):
    manifest = validate_bundle(path, "character_corpus", identity=identity)
    validate_corpus_records(path)
    return manifest


def validate_corpus_records(path):
    """Validate corpus semantics, also when embedded in prepared data."""
    rows = read_jsonl(Path(path) / "corpus.jsonl")
    inventory = {r["label"] for r in read_jsonl(Path(path) / "inventory.jsonl")}
    ids, texts, sources = [], set(), set()
    for row in rows:
        text, labels, sids = row["text"], row["labels"], row["source_ids"]
        if (not text or any(not is_han(c) for c in text) or len(text) != len(labels)
                or row["sentence_length"] != len(text) or row["positions"] != list(range(len(text)))):
            raise ValueError("corpus alignment or character contract mismatch")
        for label in labels:
            validate_canonical_label(label)
        if not set(labels).issubset(inventory) or row["sample_id"] != sample_id(text, labels):
            raise ValueError("corpus labels or identity mismatch")
        if text in texts or not sids or sids != sorted(set(sids)) or sources.intersection(sids):
            raise ValueError("duplicated corpus text or source identity")
        texts.add(text)
        sources.update(sids)
        ids.append(row["sample_id"])
    if not ids or ids != sorted(set(ids)):
        raise ValueError("corpus sample IDs must be nonempty, sorted and unique")
    stats = json.loads((Path(path) / "statistics.json").read_text(encoding="utf-8"))
    rejections = read_jsonl(Path(path) / "rejections.jsonl")
    if (stats.get("selected") != len(rows) or stats.get("rejected") != len(rejections)
            or stats.get("input") != len(rows) + len(rejections) + stats.get("duplicates", -1)):
        raise ValueError("corpus population statistics mismatch")
