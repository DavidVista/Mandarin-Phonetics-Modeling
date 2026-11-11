"""Compute contextual distances and publish an encoder geometry report."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file, write_json
from ..bundles import content_hash, seal_bundle, staging_directory, validate_bundle
from ..evaluation.encoder_extraction import load_export, load_positions
from ..evaluation.reference_geometry import evaluate_geometry
from ..evaluation.training_summary import TRAINING_MODELS, TRAINING_SEEDS
from ..visualization.encoder_geometry import rank_metrics, write_view, write_page
from ..visualization.reference_geometry import render_distance_maps
from .reference_geometry import _read_reference

SYSTEMS = ("DIMSIM", *TRAINING_MODELS)
FIGURES = tuple((f"encoder_{model.replace('-', '_')}_cosine_distances", f"{model}-cosine-distances")
                for model in TRAINING_MODELS)


def contextual_distances(labels, occurrences, hidden):
    """Equal-character, then equal-occurrence, weighted cosine distance."""
    hidden = np.asarray(hidden, dtype=np.float64)
    norms = np.linalg.norm(hidden, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0) or len(hidden) != len(occurrences):
        raise ValueError("invalid or unaligned hidden vectors")
    normalized = hidden / norms[:, None]
    groups = defaultdict(lambda: defaultdict(list))
    for index, occurrence in enumerate(occurrences):
        groups[occurrence["pinyin"]][occurrence["character"]].append(index)
    means, diagonal = [], []
    for label in labels:
        characters = groups[label]
        if not characters:
            raise ValueError(f"no occurrences for {label}")
        k = len(characters)
        mean = np.mean([normalized[indexes].mean(axis=0) for indexes in characters.values()], axis=0)
        means.append(mean)
        count = sum(map(len, characters.values()))
        square_mass = sum(1 / (k * k * len(indexes)) for indexes in characters.values())
        diagonal.append(np.clip((1 - mean @ mean) / (1 - square_mass), 0, 2) if count > 1 else np.nan)
    means = np.stack(means)
    similarities = means @ means.T
    distances = 1 - np.clip((similarities + similarities.T) / 2, -1, 1)
    np.fill_diagonal(distances, diagonal)
    return distances


def _summarize_seed_tables(tables):
    dimensions = {
        "controlled_match": ("system", "from_mask", "to_mask"),
        "dimsim_agreement": ("system", "slice"), "anchor_win_rates": ("system", "anchor"),
        "anchor_summary": ("system",), "pairwise_spearman": ("system",),
        "mask_profiles": ("system", "mask"), "mask_summary": ("system", "includes_full_match"),
        "full_match_gains": ("system", "component", "step_1", "step_2", "step_3"),
    }
    result = {}
    for name, keys in dimensions.items():
        groups = defaultdict(list)
        for row in tables[name]:
            groups[tuple(row[key] for key in keys)].append(row)
        rows = []
        for key, entries in groups.items():
            if [row["seed"] for row in entries] != list(TRAINING_SEEDS):
                raise ValueError(f"seed inventory differs for {name}")
            output = dict(zip(keys, key))
            for field in sorted(set(entries[0]) - set(keys) - {"seed"}):
                values = [row[field] for row in entries if row[field] is not None]
                output[f"{field}_mean"] = float(np.mean(values)) if values else None
                output[f"{field}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                output[f"{field}_seeds"] = len(values)
            rows.append(output)
        result[name] = rows
    return result


def _write_tables(root, tables, suffix):
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ["seed", "system", "rank", "left", "right", "mask", "residual"]
        with (root / f"{name}{suffix}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run_encoder_geometry(*, evaluation_root, pins, exports, output, tolerance, source_sha256):
    _, occurrences = load_positions(evaluation_root, pins["evaluation"])
    labels, components, reference, identity = _read_reference(evaluation_root, pins["evaluation"])
    if identity["experiment_manifest_sha256"] != pins["experiment"]:
        raise ValueError("evaluation and experiment pins differ")
    positions = [row["occurrence_id"] for row in occurrences]
    export_pins = {model: sha256_file(Path(exports[model]) / "manifest.json") for model in TRAINING_MODELS}
    report_identity = {"inputs": pins, "exports": export_pins, "tolerance": tolerance,
                       "source_sha256": source_sha256, "models": list(TRAINING_MODELS),
                       "seeds": list(TRAINING_SEEDS), "unknown_policy": "retain"}
    target = Path(output).resolve()
    for path in (evaluation_root, *exports.values()):
        root = Path(path).resolve()
        if target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("encoder report output must be separate from inputs")
    if target.exists():
        validate_bundle(target, "encoder_geometry_report", identity=report_identity)
        return target
    generators = {
        model: load_export(exports[model], model=model, evaluation_pin=pins["evaluation"],
                           experiment_pin=pins["experiment"], training_pin=pins[model], occurrence_ids=positions)
        for model in TRAINING_MODELS
    }
    paired = {}
    unknown_by_seed = {}
    all_tables = defaultdict(list)
    coverage = []
    matrix_sum = None
    map_sum = {}
    with staging_directory(target) as stage:
        for name in ("tables", "data", "figures", "views"):
            (stage / name).mkdir()
        write_json(stage / "config.json", report_identity)
        write_json(stage / "data/inventory.json", {"labels": labels, "components": [components[x] for x in labels]})
        for seed in TRAINING_SEEDS:
            distances = [reference]
            for model in TRAINING_MODELS:
                index, hidden, unknown, info = next(generators[model])
                hashes = info["seed_input_hashes"]
                if index != seed or hashes["test"] != identity["test_ids_sha256"] or paired.setdefault(seed, hashes) != hashes:
                    raise ValueError("encoder exports use different paired seed inputs")
                if not np.array_equal(unknown_by_seed.setdefault(seed, unknown), unknown):
                    raise ValueError("encoder exports disagree on paired seed UNK positions")
                distance = contextual_distances(labels, occurrences, hidden)
                distances.append(distance)
                coverage.append({"seed": seed, "system": model, "syllables": len(labels),
                                 "occurrences": len(positions), "unknown": int(unknown.sum()),
                                 "unknown_fraction": float(unknown.mean()),
                                 "planned_full_matches": len(labels),
                                 "valid_full_matches": int(np.isfinite(distance.diagonal()).sum())})
            distances = np.stack(distances)
            tables, maps = evaluate_geometry(labels, [components[x] for x in labels], distances,
                                             tolerance=tolerance, systems=SYSTEMS, contextual=True)
            rankings = rank_metrics(distances, maps["neighborhood"], tolerance)
            for row, overlap, near_far in zip(tables["pairwise_spearman"], *rankings):
                row.update(top_overlap=overlap, near_far_win_rate=near_far)
            np.savez_compressed(stage / f"data/seed-{seed}.npz", distances=distances, **maps)
            for name, rows in tables.items():
                all_tables[name].extend({"seed": seed, **row} for row in rows)
            write_view(stage / f"views/seed-{seed}.js", f"seed-{seed}", distances, maps, tables,
                       tolerance, rankings=rankings)
            if matrix_sum is None:
                matrix_sum = distances.copy()
                map_sum = {name: value.copy() for name, value in maps.items()}
            else:
                matrix_sum += distances
                for name in ("controlled_values", "agreement"):
                    map_sum[name] += maps[name]
                for name in ("controlled_cells", "agreement_counts", "component_order", "neighborhood", "masks"):
                    if not np.array_equal(map_sum[name], maps[name]):
                        raise ValueError(f"shared seed comparison support differs: {name}")
        if any(sha256_file(Path(exports[model]) / "manifest.json") != export_pins[model] for model in TRAINING_MODELS):
            raise ValueError("encoder export manifests changed during the report")
        means = matrix_sum / len(TRAINING_SEEDS)
        for name in ("controlled_values", "agreement"):
            map_sum[name] /= len(TRAINING_SEEDS)
        summary = _summarize_seed_tables(all_tables)
        np.savez_compressed(stage / "data/across-seeds.npz", distances=means, **map_sum)
        _write_tables(stage / "tables", {**all_tables, "coverage": coverage}, "")
        _write_tables(stage / "tables", summary, "_across_seeds")
        rankings = tuple([row[field] for row in summary["pairwise_spearman"]]
                         for field in ("top_overlap_mean", "near_far_win_rate_mean"))
        write_view(stage / "views/across-seeds.js", "across-seeds", means, map_sum, summary,
                   tolerance, rankings=rankings)
        write_page(stage / "encoder_geometry.html", labels, [components[x] for x in labels], coverage,
                   evaluation=identity["evaluation_id"], excluded=identity["excluded_pinyin"],
                   tolerance=tolerance)
        render_distance_maps(stage / "figures", labels, means[1:], map_sum["component_order"], figures=FIGURES,
                             provenance={"source_digest": content_hash(report_identity),
                                         "evaluation_manifest_sha256": pins["evaluation"],
                                         "hidden_export_manifests": export_pins,
                                         "aggregation": "mean distance across seeds"})
        seal_bundle(stage, "encoder_geometry_report", report_identity)
    return target
