"""Create and validate immutable, model-independent evaluation bundles.

The on-disk format intentionally uses JSON/JSONL so the preparation stage has
no scientific-runtime dependency. Embeddings are produced by later stages.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .dimsim import DimsimDistance
from .sampling import make_pair_plan, pair_count

BUNDLE_SCHEMA_VERSION = 2
PAIRING_POLICY = "nonzero_nonzero"
_EVALUATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_WINDOWS_RESERVED_NAME_PATTERN = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.IGNORECASE)
DistanceFunction = Callable[[str, str], float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _safe_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_evaluation_id(value: Any) -> str:
    if (not isinstance(value, str) or _EVALUATION_ID_PATTERN.fullmatch(value) is None
            or ".." in value or "/" in value or "\\" in value or ":" in value
            or value.endswith((".", " ")) or Path(value).is_absolute()
            or _WINDOWS_RESERVED_NAME_PATTERN.fullmatch(value) is not None):
        raise ValueError(
            "evaluation_id must be a safe single-directory name using ASCII letters, digits, '.', '_', or '-'"
        )
    return value


def _occurrence_id(sample_id: str, position: int) -> str:
    return f"{sample_id}:{position}"


def _group_id(kind: str, values: tuple[str, ...]) -> str:
    payload = json.dumps([kind, *values], ensure_ascii=False, separators=(",", ":"))
    return f"{kind}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _load_source_experiment(root: Path, split: str, sentence_limit: int | None, sentence_seed: int) -> tuple[list[dict], dict, dict, dict]:
    files = {name: root / name for name in ("manifest.json", "samples.jsonl", "splits.json", "vocab.json", "config.json")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = _read_json(files["manifest.json"])
    artifacts = manifest.get("artifacts", {})
    for name in ("samples.jsonl", "splits.json", "vocab.json", "config.json"):
        expected = artifacts.get(name)
        if not isinstance(expected, str) or _sha256(files[name]) != expected:
            raise ValueError(f"source experiment artifact checksum mismatch: {name}")
    splits, vocab = _read_json(files["splits.json"]), _read_json(files["vocab.json"])
    if split not in splits or not isinstance(splits[split], list) or not splits[split]:
        raise ValueError(f"unknown experiment split: {split}")
    if sentence_limit is not None and (isinstance(sentence_limit, bool) or not isinstance(sentence_limit, int) or sentence_limit <= 0):
        raise ValueError("sentence_limit must be positive or None")
    if any(not isinstance(values, list) for values in splits.values()):
        raise ValueError("source experiment splits must contain ID lists")
    all_split_ids = [sample_id for values in splits.values() for sample_id in values]
    if any(not isinstance(value, str) or not value for value in all_split_ids) or len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("source experiment splits are not disjoint unique ID lists")
    selected_ids = sorted(splits[split])
    if sentence_limit is not None:
        if sentence_limit > len(selected_ids):
            raise ValueError(f"sentence_limit {sentence_limit} exceeds split size {len(selected_ids)}")
        selected_ids = sorted(random.Random(sentence_seed).sample(selected_ids, sentence_limit))
    sample_rows = _read_jsonl(files["samples.jsonl"])
    samples = {row["sample_id"]: row for row in sample_rows}
    if len(samples) != len(sample_rows):
        raise ValueError("duplicate sample IDs in source experiment")
    try:
        selected = [samples[sample_id] for sample_id in selected_ids]
    except KeyError as error:
        raise ValueError(f"split references missing sample: {error.args[0]}") from error
    hashes = {name: _sha256(files[name]) for name in ("manifest.json", "samples.jsonl", "splits.json", "vocab.json", "config.json")}
    return selected, vocab, manifest, hashes


def _build_occurrences(samples: list[dict], vocab: dict) -> list[dict]:
    acoustic_tokens = vocab.get("acoustic_tokens")
    labels = vocab.get("labels", {})
    if not isinstance(acoustic_tokens, list) or any(not isinstance(x, str) for x in acoustic_tokens):
        raise ValueError("invalid acoustic token vocabulary")
    component_names = [labels.get(name) for name in ("initials", "finals", "tones")]
    if any(not isinstance(values, list) or any(not isinstance(value, str) for value in values) for values in component_names):
        raise ValueError("invalid component label vocabulary")
    rows = []
    for sample in sorted(samples, key=lambda row: row["sample_id"]):
        sample_id, text = _safe_id(sample.get("sample_id"), "sample_id"), sample.get("text")
        acoustic_ids, components = sample.get("acoustic_ids"), sample.get("component_labels")
        if not isinstance(text, str) or not (len(text) == len(acoustic_ids or []) == len(components or [])):
            raise ValueError(f"source sample shape mismatch: {sample_id}")
        for position, character in enumerate(text):
            acoustic_id = acoustic_ids[position]
            component_ids = components[position]
            if not isinstance(acoustic_id, int) or acoustic_id < 0 or acoustic_id >= len(acoustic_tokens):
                raise ValueError(f"invalid acoustic ID in sample {sample_id}")
            if not isinstance(component_ids, list) or len(component_ids) != 3:
                raise ValueError(f"invalid component labels in sample {sample_id}")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(component_names[i])
                   for i, value in enumerate(component_ids)):
                raise ValueError(f"component ID outside vocabulary in sample {sample_id}")
            try:
                initial, final, tone = (component_names[i][value] for i, value in enumerate(component_ids))
            except (IndexError, TypeError) as error:
                raise ValueError(f"component ID outside vocabulary in sample {sample_id}") from error
            pinyin = acoustic_tokens[acoustic_id]
            rows.append({
                "occurrence_id": _occurrence_id(sample_id, position), "sample_id": sample_id,
                "character_position": position, "character": character, "pinyin": pinyin,
                "initial": initial, "final": final, "tone": tone,
            })
    return sorted(rows, key=lambda row: row["occurrence_id"])


def _build_groups(occurrences: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    members: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, dict] = {}
    for row in occurrences:
        definitions = (
            ("character_pinyin", (row["character"], row["pinyin"])),
            ("syllable", (row["pinyin"],)),
        )
        for kind, keys in definitions:
            group_id = _group_id(kind, keys)
            members[group_id].append(row["occurrence_id"])
            metadata[group_id] = {
                "group_id": group_id, "group_kind": kind, "keys": list(keys),
                "pinyin": row["pinyin"], "initial": row["initial"], "final": row["final"], "tone": row["tone"],
            }
    groups = []
    for group_id in sorted(members):
        groups.append({**metadata[group_id], "occurrence_ids": sorted(members[group_id]), "occurrence_count": len(members[group_id])})
    return groups, {row["group_id"]: row for row in groups}


def _pair_rows(
    ids: list[str], metadata: dict[str, dict], *, sample_count: int | None, seed: int,
    max_all_pairs: int, distance_fn: DistanceFunction | None,
) -> list[dict]:
    rows = []
    for left_id, right_id in make_pair_plan(ids, sample_count=sample_count, seed=seed, max_all_pairs=max_all_pairs):
        left, right = metadata[left_id], metadata[right_id]
        status, distance = "not_computed", None
        if distance_fn is not None:
            try:
                distance = float(distance_fn(left["pinyin"], right["pinyin"]))
                if distance != distance or distance in (float("inf"), float("-inf")):
                    raise ValueError("distance is not finite")
                status = "ok"
            except Exception as error:  # metric coverage failures are data, not dropped rows
                status = f"invalid:{type(error).__name__}"
                distance = None
        equal_initial = left["initial"] == right["initial"]
        equal_final = left["final"] == right["final"]
        equal_tone = left["tone"] == right["tone"]
        pair_id = hashlib.sha256(json.dumps([left_id, right_id], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        rows.append({
            "pair_id": pair_id, "left_id": left_id, "right_id": right_id,
            "left_pinyin": left["pinyin"], "right_pinyin": right["pinyin"],
            "dimsim_distance": distance, "distance_status": status,
            "equal_initial": equal_initial, "equal_final": equal_final, "equal_tone": equal_tone,
            "overlap_mask": f"{int(equal_initial)}{int(equal_final)}{int(equal_tone)}",
            "left_zero_initial": left["initial"] == "", "right_zero_initial": right["initial"] == "",
            "no_zero_initial_diagnostic": left["initial"] != "" and right["initial"] != "",
        })
    return rows


def prepare_evaluation_bundle(
    evaluation_id: str, *, experiment_path: str | Path, output_root: str | Path,
    split: str = "test", sentence_limit: int | None = None, contextual_pair_count: int = 12_000,
    sentence_seed: int = 42, sampling_seed: int = 42, character_pair_count: int = 12_000,
    syllable_pair_count: int | None = None, max_all_pairs: int = 1_000_000,
    distance_fn: DistanceFunction | None = None, metric_provenance: dict[str, Any] | None = None,
    code_provenance: dict[str, str] | None = None,
) -> Path:
    """Create an immutable evaluation bundle and return its path."""
    evaluation_id = _validate_evaluation_id(evaluation_id)
    numeric = {"contextual_pair_count": contextual_pair_count, "sentence_seed": sentence_seed,
               "sampling_seed": sampling_seed, "character_pair_count": character_pair_count,
               "max_all_pairs": max_all_pairs}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in numeric.values()):
        raise TypeError("pair counts, caps, and seeds must be integers")
    if contextual_pair_count < 0 or character_pair_count < 0 or max_all_pairs <= 0:
        raise ValueError("pair counts must be non-negative and max_all_pairs positive")
    if syllable_pair_count is not None and (isinstance(syllable_pair_count, bool) or not isinstance(syllable_pair_count, int) or syllable_pair_count < 0):
        raise ValueError("syllable_pair_count must be a non-negative integer or None")
    if distance_fn is None or not callable(distance_fn):
        raise ValueError("an explicit validated distance provider is required")
    if not isinstance(metric_provenance, dict) or not isinstance(metric_provenance.get("metric_name"), str):
        raise ValueError("metric_provenance with an explicit metric_name is required")
    metric_name = metric_provenance["metric_name"]
    if metric_name == "dimsim":
        if not isinstance(distance_fn, DimsimDistance) or metric_provenance != distance_fn.provenance:
            raise ValueError("DIMSIM bundles require the validated pinned DimsimDistance provider and its provenance")
    elif metric_name != "synthetic":
        raise ValueError("only the pinned DIMSIM provider or an explicitly labelled synthetic test provider is supported")
    if not isinstance(code_provenance, dict) or not code_provenance or any(
        not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for key, value in code_provenance.items()
    ):
        raise ValueError("explicit non-empty preparation code_provenance hashes are required")
    experiment_root, root = Path(experiment_path).resolve(), Path(output_root).resolve()
    target = (root / evaluation_id).resolve()
    if target.parent != root:
        raise ValueError("evaluation path escapes output_root")
    if target == experiment_root or experiment_root in target.parents:
        raise ValueError("evaluation output must not be inside the source experiment")
    if target.exists():
        raise FileExistsError(f"evaluation bundle already exists: {target}")
    samples, vocab, source_manifest, source_hashes = _load_source_experiment(experiment_root, split, sentence_limit, sentence_seed)
    occurrences = _build_occurrences(samples, vocab)
    groups, group_by_id = _build_groups(occurrences)
    occurrence_by_id = {row["occurrence_id"]: row for row in occurrences}
    eligible_ids = {
        "contextual": sorted(row["occurrence_id"] for row in occurrences if isinstance(row["initial"], str) and row["initial"]),
        "character_pinyin": sorted(row["group_id"] for row in groups if row["group_kind"] == "character_pinyin" and isinstance(row["initial"], str) and row["initial"]),
        "syllable": sorted(row["group_id"] for row in groups if row["group_kind"] == "syllable" and isinstance(row["initial"], str) and row["initial"]),
    }
    contextual = _pair_rows(
        eligible_ids["contextual"], occurrence_by_id, sample_count=contextual_pair_count,
        seed=sampling_seed, max_all_pairs=max_all_pairs, distance_fn=distance_fn,
    )
    aggregate_plans = {}
    pair_counts = {"character_pinyin": character_pair_count, "syllable": syllable_pair_count}
    for offset, kind in enumerate(("character_pinyin", "syllable"), start=1):
        aggregate_plans[kind] = _pair_rows(
            eligible_ids[kind], group_by_id, sample_count=pair_counts[kind], seed=sampling_seed + offset,
            max_all_pairs=max_all_pairs, distance_fn=distance_fn,
        )
    target.mkdir(parents=True)
    try:
        (target / "pairs").mkdir()
        selected_sentences = [{"sample_id": row["sample_id"], "text": row["text"]} for row in sorted(samples, key=lambda x: x["sample_id"])]
        _write_jsonl(target / "sentences.jsonl", selected_sentences)
        _write_jsonl(target / "occurrences.jsonl", occurrences)
        _write_jsonl(target / "groups.jsonl", groups)
        _write_jsonl(target / "pairs/contextual.jsonl", contextual)
        for kind, rows in aggregate_plans.items():
            _write_jsonl(target / f"pairs/{kind}.jsonl", rows)
        object_counts = {
            "contextual": len(occurrences),
            "character_pinyin": sum(row["group_kind"] == "character_pinyin" for row in groups),
            "syllable": sum(row["group_kind"] == "syllable" for row in groups),
        }
        all_plans = {"contextual": contextual, **aggregate_plans}
        coverage = {
            "sentences": len(samples), "occurrences": len(occurrences),
            "groups": {kind: object_counts[kind] for kind in ("character_pinyin", "syllable")},
            "levels": {},
        }
        for level, rows in all_plans.items():
            statuses: dict[str, int] = {}
            for row in rows:
                statuses[row["distance_status"]] = statuses.get(row["distance_status"], 0) + 1
            original_count = object_counts[level]
            eligible_count = len(eligible_ids[level])
            original_pairs, eligible_pairs = pair_count(original_count), pair_count(eligible_count)
            coverage["levels"][level] = {
                "objects": {
                    "original": original_count, "eligible": eligible_count,
                    "excluded": original_count - eligible_count,
                    "excluded_by_reason": {"zero_initial": original_count - eligible_count},
                },
                "unordered_pair_population": {
                    "original": original_pairs, "eligible": eligible_pairs,
                    "excluded": original_pairs - eligible_pairs,
                    "excluded_by_reason": {"zero_initial": original_pairs - eligible_pairs},
                },
                "selection_count": len(rows),
                "distance_status_counts": dict(sorted(statuses.items())),
            }
        _write_json(target / "coverage.json", coverage)
        config = {
            "schema_version": BUNDLE_SCHEMA_VERSION, "evaluation_id": evaluation_id, "split": split,
            "sentence_limit": sentence_limit, "contextual_pair_count": contextual_pair_count,
            "sentence_seed": sentence_seed, "sampling_seed": sampling_seed,
            "character_pair_count": character_pair_count, "syllable_pair_count": syllable_pair_count,
            "max_all_pairs": max_all_pairs,
            "pairing_policy": PAIRING_POLICY,
        }
        _write_json(target / "config.json", config)
        artifacts = {str(path.relative_to(target)).replace("\\", "/"): _sha256(path) for path in target.rglob("*") if path.is_file()}
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION, "evaluation_id": evaluation_id,
            "pairing_policy": PAIRING_POLICY,
            "source_experiment": {
                "path": str(experiment_root), "experiment_id": source_manifest.get("experiment_id"),
                "hashes": source_hashes,
            },
            "metric": metric_provenance,
            "code_provenance": code_provenance,
            "counts": {
                "sentences": len(samples), "occurrences": len(occurrences),
                "character_pinyin_groups": sum(row["group_kind"] == "character_pinyin" for row in groups),
                "syllable_groups": sum(row["group_kind"] == "syllable" for row in groups),
                "pairs": {"contextual": len(contextual), **{kind: len(rows) for kind, rows in aggregate_plans.items()}},
            },
            "artifacts": artifacts,
        }
        _write_json(target / "manifest.json", manifest)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def load_evaluation_bundle(path: str | Path) -> dict[str, Any]:
    """Load a bundle after verifying every recorded artifact checksum."""
    root = Path(path).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation bundle schema")
    config = _read_json(root / "config.json")
    if config.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation config schema")
    if manifest.get("pairing_policy") != PAIRING_POLICY or config.get("pairing_policy") != PAIRING_POLICY:
        raise ValueError("unsupported evaluation pairing policy")
    manifest_evaluation_id = _validate_evaluation_id(manifest.get("evaluation_id"))
    config_evaluation_id = _validate_evaluation_id(config.get("evaluation_id"))
    if manifest_evaluation_id != config_evaluation_id:
        raise ValueError("invalid or inconsistent evaluation_id")
    required = {"config.json", "coverage.json", "sentences.jsonl", "occurrences.jsonl", "groups.jsonl",
                "pairs/contextual.jsonl", "pairs/character_pinyin.jsonl", "pairs/syllable.jsonl"}
    artifacts = manifest.get("artifacts", {})
    if not required.issubset(artifacts):
        raise ValueError("evaluation manifest is missing required artifacts")
    for relative, expected in artifacts.items():
        artifact = (root / relative).resolve()
        if root not in artifact.parents or not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"evaluation artifact checksum mismatch: {relative}")
    coverage = _read_json(root / "coverage.json")
    occurrences, groups = _read_jsonl(root / "occurrences.jsonl"), _read_jsonl(root / "groups.jsonl")
    occurrence_ids = [row.get("occurrence_id") for row in occurrences]
    group_ids = [row.get("group_id") for row in groups]
    if len(occurrence_ids) != len(set(occurrence_ids)) or any(not isinstance(value, str) for value in occurrence_ids):
        raise ValueError("evaluation occurrence IDs are invalid or duplicated")
    if len(group_ids) != len(set(group_ids)) or any(not isinstance(value, str) for value in group_ids):
        raise ValueError("evaluation group IDs are invalid or duplicated")
    if any(not isinstance(row.get("initial"), str) for row in occurrences + groups):
        raise ValueError("evaluation initial metadata must be strings")
    pairs = {kind: _read_jsonl(root / f"pairs/{kind}.jsonl") for kind in ("contextual", "character_pinyin", "syllable")}
    occurrence_id_set = set(occurrence_ids)
    valid_ids = {
        "contextual": occurrence_id_set,
        "character_pinyin": {row["group_id"] for row in groups if row.get("group_kind") == "character_pinyin"},
        "syllable": {row["group_id"] for row in groups if row.get("group_kind") == "syllable"},
    }
    metadata = {"contextual": {row["occurrence_id"]: row for row in occurrences},
                "character_pinyin": {row["group_id"]: row for row in groups if row.get("group_kind") == "character_pinyin"},
                "syllable": {row["group_id"]: row for row in groups if row.get("group_kind") == "syllable"}}
    for group in groups:
        members = group.get("occurrence_ids")
        if (not isinstance(members, list) or not members or members != sorted(members) or len(members) != len(set(members))
                or group.get("occurrence_count") != len(members) or any(member not in occurrence_id_set for member in members)):
            raise ValueError(f"invalid group membership: {group.get('group_id')}")
    for kind, rows in pairs.items():
        pair_ids = [row.get("pair_id") for row in rows]
        if len(pair_ids) != len(set(pair_ids)) or any(not isinstance(value, str) for value in pair_ids):
            raise ValueError(f"{kind} pair IDs are invalid or duplicated")
        if any(row.get("left_id") not in valid_ids[kind] or row.get("right_id") not in valid_ids[kind] for row in rows):
            raise ValueError(f"{kind} pair references an unknown object")
        endpoints = [(row["left_id"], row["right_id"]) for row in rows]
        if len(endpoints) != len(set(endpoints)) or any(left >= right for left, right in endpoints):
            raise ValueError(f"{kind} pairs are duplicated or non-canonical")
        for row in rows:
            expected = hashlib.sha256(json.dumps([row["left_id"], row["right_id"]], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if row["pair_id"] != expected:
                raise ValueError(f"{kind} pair ID does not match endpoints")
            left_zero = metadata[kind][row["left_id"]].get("initial") == ""
            right_zero = metadata[kind][row["right_id"]].get("initial") == ""
            if left_zero or right_zero:
                raise ValueError(f"{kind} pair violates {PAIRING_POLICY} policy")
            if row.get("left_zero_initial") is not False or row.get("right_zero_initial") is not False or row.get("no_zero_initial_diagnostic") is not True:
                raise ValueError(f"{kind} pair has inconsistent zero-initial flags")
        original_count = len(valid_ids[kind])
        eligible_count = sum(isinstance(metadata[kind][object_id].get("initial"), str) and bool(metadata[kind][object_id]["initial"])
                             for object_id in valid_ids[kind])
        original_pairs, eligible_pairs = pair_count(original_count), pair_count(eligible_count)
        statuses: dict[str, int] = {}
        for row in rows:
            status = row.get("distance_status")
            statuses[status] = statuses.get(status, 0) + 1
        expected_coverage = {
            "objects": {
                "original": original_count, "eligible": eligible_count, "excluded": original_count - eligible_count,
                "excluded_by_reason": {"zero_initial": original_count - eligible_count},
            },
            "unordered_pair_population": {
                "original": original_pairs, "eligible": eligible_pairs, "excluded": original_pairs - eligible_pairs,
                "excluded_by_reason": {"zero_initial": original_pairs - eligible_pairs},
            },
            "selection_count": len(rows), "distance_status_counts": dict(sorted(statuses.items())),
        }
        if coverage.get("levels", {}).get(kind) != expected_coverage:
            raise ValueError(f"{kind} coverage metadata is inconsistent")
    expected_counts = {
        "sentences": len(_read_jsonl(root / "sentences.jsonl")), "occurrences": len(occurrences),
        "character_pinyin_groups": len(valid_ids["character_pinyin"]), "syllable_groups": len(valid_ids["syllable"]),
        "pairs": {kind: len(rows) for kind, rows in pairs.items()},
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("evaluation manifest counts are inconsistent")
    return {
        "root": root, "manifest": manifest, "config": config,
        "coverage": coverage,
        "sentences": _read_jsonl(root / "sentences.jsonl"),
        "occurrences": occurrences, "groups": groups, "pairs": pairs,
    }
