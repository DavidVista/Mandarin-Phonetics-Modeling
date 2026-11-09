"""Publish a reference geometry report from authenticated local bundles."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from ..artifacts import safe_artifact, sha256_file, write_json
from ..bundles import content_hash
from ..evaluation.geometry_bundle import validate_geometry_bundle
from ..evaluation.graduality_io import load_embedding_bundle
from ..evaluation.reference_geometry import SYSTEMS, SLICES, TRANSITIONS, evaluate_geometry
from ..pinyin import parse_components
from ..visualization.reference_geometry import render_distance_maps, render_page

_MODELS = {
    "base_root": "facebook/wav2vec2-base",
    "zh_cn_root": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}


def _read_reference(root, manifest_pin):
    root = Path(root)
    if sha256_file(safe_artifact(root, "manifest.json")) != manifest_pin:
        raise ValueError("evaluation manifest SHA256 mismatch")
    manifest = validate_geometry_bundle(root)
    group_bytes = safe_artifact(root, "groups.jsonl").read_bytes()
    if hashlib.sha256(group_bytes).hexdigest() != manifest["artifacts"]["groups.jsonl"]:
        raise ValueError("evaluation groups differ from the pinned bundle")
    groups = [json.loads(line) for line in group_bytes.splitlines()]
    identity = manifest["identity"]
    by_label = {row["pinyin"]: row for row in groups}
    labels = sorted(by_label)
    if labels != identity["inventory"]["eligible"] or len(labels) != len(groups):
        raise ValueError("evaluation groups disagree with the sealed inventory")
    components = {}
    for label in labels:
        parsed = parse_components(label)
        fields = (parsed.initial, parsed.final, parsed.tone)
        if not parsed.initial or fields != tuple(by_label[label][key] for key in ("initial", "final", "tone")):
            raise ValueError(f"invalid evaluation components for {label}")
        components[label] = fields
    distances = np.zeros((len(labels), len(labels)), dtype=np.float64)
    indexes = {label: index for index, label in enumerate(labels)}
    endpoints = iter(itertools.combinations(sorted((row["group_id"], row["pinyin"]) for row in groups), 2))
    count = 0
    digest = hashlib.sha256()
    with safe_artifact(root, "pairs/syllable.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            pair = json.loads(line)
            expected = next(endpoints, None)
            if expected != ((pair["left_id"], pair["left_pinyin"]), (pair["right_id"], pair["right_pinyin"])):
                raise ValueError("evaluation pair endpoints are incomplete or out of order")
            i, j = indexes[pair["left_pinyin"]], indexes[pair["right_pinyin"]]
            value = pair["dimsim_distance"]
            if pair["distance_status"] != "ok" or type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid stored DIMSIM distance")
            distances[i, j] = distances[j, i] = value
            count += 1
    if (next(endpoints, None) is not None or count != identity["syllable_pair_count"]
            or digest.hexdigest() != manifest["artifacts"]["pairs/syllable.jsonl"]
            or sha256_file(safe_artifact(root, "manifest.json")) != manifest_pin):
        raise ValueError("evaluation pair table or manifest differs from the pinned bundle")
    return labels, components, distances, identity


def _cosine_distances(labels, vectors):
    matrix = np.stack([vectors[label] for label in labels])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    distances = 1 - np.clip(matrix @ matrix.T, -1, 1)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    return distances


def run_reference_geometry(root, parameters, pins):
    root = Path(root)
    tolerance = parameters.get("tolerance", 1e-12)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("reference geometry tolerance must be finite and nonnegative")
    labels, components, dimsim, identity = _read_reference(parameters["evaluation_root"], pins["evaluation_root"])
    acoustic = {key: load_embedding_bundle(key, parameters[key], pins[key], model) for key, model in _MODELS.items()}
    missing = {key: sorted(set(labels) - bundle["vectors"].keys()) for key, bundle in acoustic.items()}
    common = sorted(set(labels) - set().union(*missing.values()))
    if len(common) < 2:
        raise ValueError("reference geometry requires at least two shared syllables")
    common_set = set(common)
    indices = [index for index, label in enumerate(labels) if label in common_set]
    audio_hashes = [{row["label"]: row["sha256"] for row in bundle["inventory"]} for bundle in acoustic.values()]
    if any(not audio_hashes[0][label] or audio_hashes[0][label] != audio_hashes[1][label] for label in common):
        raise ValueError("wav2vec2 source audio differs on the common inventory")
    distances = np.stack([dimsim[np.ix_(indices, indices)],
                          *(_cosine_distances(common, bundle["vectors"]) for bundle in acoustic.values())])
    tables, maps = evaluate_geometry(common, [components[label] for label in common], distances, tolerance=tolerance)
    n = len(common)
    coverage = [
        {"population": "Evaluation syllables", "count": len(labels)},
        {"population": "Common syllables", "count": n},
        {"population": "Missing from wav2vec2 base", "count": len(missing["base_root"])},
        {"population": "Missing from wav2vec2 zh-cn", "count": len(missing["zh_cn_root"])},
        {"population": "Unique off-diagonal pairs", "count": n * (n - 1) // 2},
        {"population": "Possible distinct triples", "count": n * (n - 1) * (n - 2) // 2},
        {"population": "Possible triples including full match", "count": n * n * (n - 1) // 2},
    ]
    for row in tables["dimsim_agreement"]:
        if row["system"] == SYSTEMS[0]:
            coverage.extend([
                {"population": f"{row['slice']} triples", "count": row["triples"]},
                {"population": f"{row['slice']} reference comparisons", "count": row["comparisons"]},
            ])
    tables = {"coverage": coverage, **tables}

    for folder in ("data", "tables", "figures"):
        (root / folder).mkdir()
    write_json(root / "config.json", {
        "inputs": pins, "systems": SYSTEMS, "exploratory": True, "tolerance": tolerance,
        "ties": "absolute tolerance, model ties receive half credit, reference ties excluded from agreement",
        "weights": "equal weight per unique planned triple or off-diagonal pair",
        "diagonal": "zero for all reference systems",
        "neighborhood_ties": "ascending distance followed by lexical Pinyin order",
        "exclusions": identity["excluded_pinyin"], "exclusion_source": identity["exclusion_source"],
        "exclusion_confirmation": identity.get("exclusion_confirmation"),
        "acoustic_models": {key: bundle["manifest"]["model"] for key, bundle in acoustic.items()},
        "dimsim_metric": identity["metric"],
    })
    write_json(root / "data/inventory.json", {
        "labels": common, "components": [components[label] for label in common],
        "missing_by_system": missing, "evaluation_inventory": labels,
    })
    np.savez_compressed(root / "data/distances.npz", distances=distances, systems=SYSTEMS)
    np.savez_compressed(root / "data/map_data.npz", **maps, transitions=TRANSITIONS,
                        systems=SYSTEMS, agreement_slices=SLICES[1:])
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ["system", "rank", "left", "right", "mask", "residual"]
        with (root / f"tables/{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    figure_inputs = {name: sha256_file(root / name) for name in ("data/distances.npz", "data/inventory.json")}
    render_distance_maps(root / "figures", common, distances, maps["component_order"], provenance={
        "source_digest": content_hash(figure_inputs), "source_artifacts": figure_inputs,
        "input_manifests": pins,
    })
    render_page(root / "reference_geometry.html", common, [components[label] for label in common],
                distances, maps, tables, evaluation_id=identity["evaluation_id"],
                excluded=identity["excluded_pinyin"], tolerance=tolerance)
