"""Build the fixed syllable geometry population from a character experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from ..artifacts import write_json, write_jsonl
from ..bundles import content_hash, seal_bundle, staging_directory, validate_bundle
from ..experiment import load_common_test
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, parse_components, validate_canonical_label
from .bundle import _group_id, _iter_pair_rows, _occurrence_id, _validate_evaluation_id
from .dimsim import DimsimDistance
from .sampling import pair_count

BUNDLE_TYPE = "syllable_geometry_evaluation"
PAIRING_POLICY = "strict_nonzero_excluded_pinyin"
ARTIFACTS = {"inventory.json", "sentences.jsonl", "occurrences.jsonl", "groups.jsonl", "pairs/syllable.jsonl"}


def _manifest_pin(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("a lowercase SHA256 manifest pin is required")
    return value


def _validate_audit_reference(reference, field):
    if (not isinstance(reference, dict) or set(reference) != {"run_id", "manifest_sha256"}
            or not isinstance(reference["run_id"], str) or not reference["run_id"].strip()):
        raise ValueError(f"{field} requires a run_id and manifest_sha256")
    _manifest_pin(reference["manifest_sha256"])


def _validate_exclusions(labels):
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError("excluded_pinyin must be a list of canonical tone-labelled forms")
    if len(labels) != len(set(labels)):
        raise ValueError("excluded_pinyin must contain unique labels")
    for label in labels:
        validate_canonical_label(label)


def resolve_evaluation_config(spec, resolve):
    fields = {"id", "experiment_root", "experiment_manifest_sha256", "output_root",
              "excluded_pinyin", "exclusion_source"}
    if not isinstance(spec, dict) or not fields <= set(spec) <= fields | {"exclusion_confirmation"}:
        raise ValueError("evaluation requires id, experiment_root, experiment_manifest_sha256, "
                         "output_root, excluded_pinyin and exclusion_source")
    _validate_evaluation_id(spec["id"])
    _manifest_pin(spec["experiment_manifest_sha256"])
    _validate_exclusions(spec["excluded_pinyin"])
    _validate_audit_reference(spec["exclusion_source"], "exclusion_source")
    if "exclusion_confirmation" in spec:
        _validate_audit_reference(spec["exclusion_confirmation"], "exclusion_confirmation")
    return {**spec, "experiment_root": resolve(spec["experiment_root"]),
            "output_root": resolve(spec["output_root"])}


def validate_geometry_bundle(path, *, identity=None):
    manifest = validate_bundle(path, BUNDLE_TYPE, identity=identity)
    if set(manifest["artifacts"]) != ARTIFACTS or manifest["identity"].get("pairing_policy") != PAIRING_POLICY:
        raise ValueError("invalid geometry evaluation bundle contract")
    return manifest


def prepare_geometry_bundle(spec):
    """Publish a sealed test population and all its unique DIMSIM pairs."""
    spec = resolve_evaluation_config(spec, lambda path: Path(path).resolve())
    experiment = spec["experiment_root"]
    output_root = spec["output_root"]
    target = output_root / spec["id"]
    if target == experiment or experiment in target.parents or target in experiment.parents:
        raise ValueError("evaluation output must be separate from the experiment")
    pin = spec["experiment_manifest_sha256"]
    test_ids, samples = load_common_test(experiment, expected_manifest_sha256=pin)

    components = {label: parse_components(label) for label in sorted({
        label for sample in samples for label in sample["labels"]
    })}
    excluded = set(spec["excluded_pinyin"])
    zero_initial = [label for label, parsed in components.items() if not parsed.initial]
    applied = [label for label, parsed in components.items() if parsed.initial and label in excluded]
    eligible = [label for label, parsed in components.items() if parsed.initial and label not in excluded]
    if len(eligible) < 2:
        raise ValueError("evaluation requires at least two eligible syllables")
    if len({(components[label].initial, components[label].final, components[label].tone)
            for label in eligible}) != len(eligible):
        raise ValueError("eligible syllables have duplicate strict components")
    inventory = {
        "observed": list(components), "eligible": eligible, "zero_initial": zero_initial,
        "excluded_by_list": applied, "configured_but_unobserved": sorted(excluded - components.keys()),
    }
    metric = DimsimDistance()
    identity = {
        "evaluation_id": spec["id"], "experiment_manifest_sha256": pin,
        "split": "test", "test_ids_sha256": content_hash(test_ids), "test_sentences": len(samples),
        "pairing_policy": PAIRING_POLICY, "excluded_pinyin": sorted(excluded),
        "exclusion_source": spec["exclusion_source"], "inventory": inventory,
        "syllable_pair_count": pair_count(len(eligible)), "pinyin_contract": PINYIN_CONTRACT_VERSION,
        "parser_provenance": PARSER_PROVENANCE, "metric": metric.provenance,
    }
    if "exclusion_confirmation" in spec:
        identity["exclusion_confirmation"] = spec["exclusion_confirmation"]
    if target.exists() or target.is_symlink():
        validate_geometry_bundle(target, identity=identity)
        return target

    groups = {}
    by_label = {}
    for label in eligible:
        parsed = components[label]
        group_id = _group_id("syllable", (label,))
        group = {"group_id": group_id, "group_kind": "syllable", "keys": [label], "pinyin": label,
                 "initial": parsed.initial, "final": parsed.final, "tone": parsed.tone}
        groups[group_id] = group
        by_label[label] = group
    sentences = []
    occurrences = []
    for sample in sorted(samples, key=lambda sample: sample["sample_id"]):
        sample_id, sentence = sample["sample_id"], sample["text"]
        sentences.append({"sample_id": sample_id, "text": sentence})
        for position, (character, label) in enumerate(zip(sentence, sample["labels"])):
            if label in by_label:
                group = by_label[label]
                occurrences.append({
                    "occurrence_id": _occurrence_id(sample_id, position), "sample_id": sample_id,
                    "character_position": position, "character": character, "pinyin": label,
                    "initial": group["initial"], "final": group["final"], "tone": group["tone"],
                })

    with staging_directory(target) as staging:
        write_json(staging / "inventory.json", inventory)
        write_jsonl(staging / "sentences.jsonl", sentences)
        write_jsonl(staging / "occurrences.jsonl", occurrences)
        write_jsonl(staging / "groups.jsonl", (groups[key] for key in sorted(groups)))
        (staging / "pairs").mkdir()
        pairs = _iter_pair_rows(sorted(groups), groups, sample_count=None, seed=0,
                                max_all_pairs=identity["syllable_pair_count"], distance_fn=metric)
        with (staging / "pairs/syllable.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for pair in pairs:
                distance = pair["dimsim_distance"]
                if pair["distance_status"] != "ok" or not math.isfinite(distance) or distance < 0:
                    raise ValueError(f"invalid DIMSIM distance for {pair['left_pinyin']} and {pair['right_pinyin']}")
                stream.write(json.dumps(pair, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        seal_bundle(staging, BUNDLE_TYPE, identity)
    return target
