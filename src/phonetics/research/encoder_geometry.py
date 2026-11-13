"""Compute contextual distances and publish an encoder geometry report."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file, write_json
from ..bundles import content_hash, seal_bundle, staging_directory, validate_bundle
from ..evaluation.encoder_extraction import expected_unknown_positions, load_export, load_positions
from ..evaluation.geometry_bundle import PAIRING_POLICY
from ..evaluation.reference_geometry import component_masks, evaluate_geometry
from ..evaluation.training_summary import TRAINING_MODELS, TRAINING_SEEDS
from ..visualization.encoder_geometry import rank_metrics, write_view, write_page
from ..visualization.reference_geometry import render_distance_maps
from .reference_geometry import _read_reference

SYSTEMS = ("DIMSIM", *TRAINING_MODELS)
BASELINE = "Component mismatch count"
CONTRASTS = (
    ("syllable", "components"),
    ("components", "components-acoustic-lambda-1"),
    ("components", "components-acoustic-lambda-10"),
)
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
        "dimsim_agreement": ("system", "slice"), "same_mask_agreement": ("system", "mask"),
        "anchor_win_rates": ("system", "anchor"),
        "anchor_summary": ("system",), "pairwise_spearman": ("system",),
        "mask_profiles": ("system", "mask"), "mask_summary": ("system", "includes_full_match"),
        "full_match_gains": ("system", "component", "step_1", "step_2", "step_3"),
        "paired_differences": ("from_system", "to_system", "metric", "slice"),
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


def _paired_differences(tables):
    rates = {(row["system"], "DIMSIM Agreement", row["slice"]): row["agreement"]
             for row in tables["dimsim_agreement"]}
    rates.update({(row["system"], "Same mask agreement", row["mask"]): row["agreement"]
                  for row in tables["same_mask_agreement"]})
    rates.update({(row["system"], "Spearman", "Distinct syllables"): row["spearman"]
                  for row in tables["pairwise_spearman"]})
    rates.update({(row["system"], "Controlled match win rate", "All"): row["win_rate"]
                  for row in tables["controlled_match"] if row["from_mask"] == "All"})
    rows = []
    for before, after in CONTRASTS:
        for (system, metric, population), value in rates.items():
            if system == before:
                other = rates[after, metric, population]
                rows.append({"from_system": before, "to_system": after, "metric": metric,
                             "slice": population, "difference": other - value
                             if other is not None and value is not None else None})
    return rows


def _write_tables(root, tables, suffix):
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ["seed", "system", "rank", "left", "right", "mask", "residual"]
        with (root / f"{name}{suffix}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run_encoder_geometry(*, evaluation_root, experiment_root, pins, exports, checkpoints,
                         output, tolerance, source_sha256):
    _, occurrences = load_positions(evaluation_root, pins["evaluation"])
    labels, components, reference, identity = _read_reference(evaluation_root, pins["evaluation"])
    if identity["experiment_manifest_sha256"] != pins["experiment"]:
        raise ValueError("evaluation and experiment pins differ")
    if identity["pairing_policy"] != PAIRING_POLICY:
        raise ValueError("encoder geometry requires the complete nonzero-initial evaluation bundle")
    positions = [row["occurrence_id"] for row in occurrences]
    unknown_by_seed, experiment_hashes = expected_unknown_positions(
        experiment_root, pins["experiment"], occurrences, identity["test_ids_sha256"]
    )
    excluded = set(identity["inventory"]["excluded_by_list"])
    indices_by_seed, counts_by_seed = {}, {}
    candidate_labels = set(labels) - excluded
    for seed, unknown in unknown_by_seed.items():
        indices_by_seed[seed] = [index for index, row in enumerate(occurrences)
                                 if not unknown[index] and row["pinyin"] in candidate_labels]
        counts = defaultdict(int)
        for index in indices_by_seed[seed]:
            counts[occurrences[index]["pinyin"]] += 1
        counts_by_seed[seed] = counts
    common = set.intersection(*(set(counts) for counts in counts_by_seed.values()))
    if len(common) < 2:
        raise ValueError("primary geometry requires at least two syllables shared across all seeds")
    label_indices = [index for index, label in enumerate(labels) if label in common]
    reference = reference[np.ix_(label_indices, label_indices)]
    labels = [labels[index] for index in label_indices]
    component_rows = [components[label] for label in labels]
    primary_occurrences = sum(row["pinyin"] in candidate_labels for row in occurrences)
    full_match = np.array([all(counts[label] > 1 for counts in counts_by_seed.values()) for label in labels])
    unavailable = np.flatnonzero(~full_match)
    mask = component_masks(component_rows)
    baseline = sum((mask & bit) == 0 for bit in (4, 2, 1)).astype(np.float64)
    baseline[unavailable, unavailable] = np.nan
    export_pins = {model: sha256_file(Path(exports[model]) / "manifest.json") for model in TRAINING_MODELS}
    report_identity = {"inputs": pins, "exports": export_pins, "tolerance": tolerance,
                       "source_sha256": source_sha256, "models": list(TRAINING_MODELS),
                       "seeds": list(TRAINING_SEEDS), "unknown_policy": "exclude_target_position",
                       "population": "common_primary_syllables_across_seeds",
                       "full_match_policy": "at_least_two_occurrences_in_every_seed",
                       "checkpoints": {model: {str(seed): item for seed, item in selections.items()}
                                       for model, selections in checkpoints.items()}}
    target = Path(output).resolve()
    for path in (evaluation_root, experiment_root, *exports.values()):
        root = Path(path).resolve()
        if target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("encoder report output must be separate from inputs")
    generators = {
        model: load_export(exports[model], model=model, evaluation_pin=pins["evaluation"],
                           experiment_pin=pins["experiment"], training_pin=pins[model], occurrence_ids=positions,
                           expected_seeds=checkpoints[model])
        for model in TRAINING_MODELS
    }
    if target.exists():
        for generator in generators.values():
            for seed, _, unknown, info in generator:
                if (info["seed_input_hashes"] != experiment_hashes[seed]
                        or not np.array_equal(unknown, unknown_by_seed[seed])):
                    raise ValueError("encoder export differs from the pinned seed inputs")
        validate_bundle(target, "encoder_geometry_report", identity=report_identity)
        return target
    all_tables = defaultdict(list)
    coverage = []
    matrix_sum = None
    map_sum = {}
    with staging_directory(target) as stage:
        for name in ("tables", "data", "figures", "views"):
            (stage / name).mkdir()
        write_json(stage / "config.json", report_identity)
        write_json(stage / "data/inventory.json", {
            "labels": labels, "components": component_rows,
            "full_match_labels": [label for label, available in zip(labels, full_match) if available],
            "lost_after_unk_by_seed": {str(seed): sorted(candidate_labels - counts.keys())
                                       for seed, counts in counts_by_seed.items()},
            "excluded_from_shared_inventory": sorted(candidate_labels - common),
        })
        for seed in TRAINING_SEEDS:
            selected = [index for index in indices_by_seed[seed] if occurrences[index]["pinyin"] in common]
            retained = [occurrences[index] for index in selected]
            distances = [reference]
            primary_unknown = primary_occurrences - len(indices_by_seed[seed])
            for model in TRAINING_MODELS:
                index, hidden, unknown, info = next(generators[model])
                if (index != seed or info["seed_input_hashes"] != experiment_hashes[seed]
                        or not np.array_equal(unknown, unknown_by_seed[seed])):
                    raise ValueError("encoder export differs from the pinned seed inputs")
                distance = contextual_distances(labels, retained, hidden[selected])
                distance[unavailable, unavailable] = np.nan
                distances.append(distance)
                coverage.append({"seed": seed, "system": model, "syllables": len(labels),
                                 "occurrences": len(retained), "exported_occurrences": len(positions),
                                 "primary_occurrences_before_unk": primary_occurrences,
                                 "unknown": primary_unknown, "unknown_fraction": primary_unknown / primary_occurrences,
                                 "syllables_lost_after_unk": len(candidate_labels - counts_by_seed[seed].keys()),
                                 "planned_full_matches": len(labels),
                                 "supported_full_matches_in_seed": sum(counts_by_seed[seed][label] > 1
                                                                        for label in labels),
                                 "valid_full_matches": int(full_match.sum())})
            distances = np.stack(distances)
            comparisons = np.concatenate((distances, baseline[None]))
            tables, maps = evaluate_geometry(labels, component_rows, comparisons,
                                             tolerance=tolerance, systems=(*SYSTEMS, BASELINE), contextual=True)
            maps["controlled_values"] = maps["controlled_values"][:, :len(SYSTEMS)]
            maps["agreement"] = maps["agreement"][:, :len(SYSTEMS)]
            for name in ("mask_profiles", "mask_summary", "full_match_gains", "residuals_top50"):
                tables[name] = [row for row in tables[name] if row["system"] != BASELINE]
            overlap_rates, near_far_rates = rank_metrics(comparisons, maps["neighborhood"], tolerance)
            for row, overlap, near_far in zip(tables["pairwise_spearman"], overlap_rates, near_far_rates):
                row.update(top_overlap=overlap, near_far_win_rate=near_far)
            rankings = (overlap_rates[:len(SYSTEMS)], near_far_rates[:len(SYSTEMS)])
            tables["paired_differences"] = _paired_differences(tables)
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
        rankings = tuple([row[field] for row in summary["pairwise_spearman"] if row["system"] in SYSTEMS]
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
