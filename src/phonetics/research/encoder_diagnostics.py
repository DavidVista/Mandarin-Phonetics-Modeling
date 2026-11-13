"""Optional sensitivity tables for the sealed encoder geometry inputs."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from ..evaluation.reference_geometry import component_masks, evaluate_geometry


def plan_diagnostics(labels, occurrences, unknown_by_seed, excluded, primary):
    counts = {}
    populations = {}
    for seed, unknown in unknown_by_seed.items():
        groups = defaultdict(set)
        totals = defaultdict(int)
        for row, is_unknown in zip(occurrences, unknown):
            if not is_unknown:
                label = row["pinyin"]
                groups[label].add(row["character"])
                totals[label] += 1
        counts[seed] = {label: (totals[label], len(groups[label])) for label in labels}
    supported = [label for label in primary
                 if all(count[label][0] > 10 and count[label][1] >= 3 for count in counts.values())]
    expanded = [label for label in labels if all(count[label][0] for count in counts.values())]
    for name, inventory, retain_unknown in (
        ("unk_retained", primary, True), ("high_support", supported, False),
        ("expanded", expanded, False),
    ):
        inventory_set = set(inventory)
        indices = [i for i, label in enumerate(labels) if label in inventory_set]
        positions = {
            seed: [i for i, row in enumerate(occurrences)
                   if row["pinyin"] in inventory_set and (retain_unknown or not unknown[i])]
            for seed, unknown in unknown_by_seed.items()
        }
        # Match the primary Full Match support for the UNK contrast.
        match = [label for label in inventory if all(count[label][0] >= 2 for count in counts.values())]
        populations[name] = (inventory, indices, positions, match)
    support = [{"seed": seed, "pinyin": label, "excluded_from_primary": label in excluded,
                "occurrences": count, "characters": characters}
               for seed, rows in counts.items() for label, (count, characters) in rows.items()]
    return populations, support


def strict_reference_agreement(components, reference_a, reference_b, models, systems, tolerance):
    """Compare both references on the same strictly ordered distinct-syllable triples."""
    masks = component_masks(components)
    left, right = np.triu_indices(len(components), 1)
    totals = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    for anchor in range(len(components)):
        distinct = (left != anchor) & (right != anchor)
        low = np.minimum(masks[anchor, left], masks[anchor, right])
        high = np.maximum(masks[anchor, left], masks[anchor, right])
        shared = low & high
        a = reference_a[anchor, right] - reference_a[anchor, left]
        b = reference_b[anchor, right] - reference_b[anchor, left]
        strict_a, strict_b = abs(a) > tolerance, abs(b) > tolerance
        for name, selection in (
            ("Controlled match", distinct & (shared == low) & (low != high)
             & ((7 ^ masks[left, right]) == (low ^ high))),
            ("Trade-off", distinct & (shared != low)), ("Same mask", distinct & (low == high)),
        ):
            valid = selection & strict_a & strict_b
            for system, matrix in zip(systems, models):
                delta = matrix[anchor, right[valid]] - matrix[anchor, left[valid]]
                ties = abs(delta) <= tolerance
                totals[name, system] += [selection.sum(), (selection & ~strict_a).sum(),
                                         (selection & ~strict_b).sum(), valid.sum(), ties.sum(),
                                         ((delta * np.sign(a[valid]) > tolerance)).sum(),
                                         ((delta * np.sign(b[valid]) > tolerance)).sum()]
    rows = []
    for (name, system), (planned, ties_a, ties_b, compared, ties, wins_a, wins_b) in totals.items():
        rows.append({"slice": name, "system": system, "triples": planned, "dimsim_a_ties": ties_a,
                     "dimsim_b_ties": ties_b, "comparisons": compared, "model_ties": ties,
                     "agreement_a": (wins_a + .5 * ties) / compared if compared else None,
                     "agreement_b": (wins_b + .5 * ties) / compared if compared else None})
    return rows


def full_match_rows(labels, components, occurrences, hidden, distances, tolerance, system):
    """Compare the two diagonal estimators against the same one-component candidates."""
    groups = defaultdict(lambda: defaultdict(list))
    vectors = hidden.astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for row, vector in zip(occurrences, vectors):
        groups[row["pinyin"]][row["character"]].append(vector)
    masks = component_masks(components)
    rows = []
    for index, label in enumerate(labels):
        groups_for_label = groups[label]
        if len(groups_for_label) < 2:
            continue
        means = [np.mean(items, axis=0) for items in groups_for_label.values()]
        pair_distances = [1 - np.clip(a @ b, -1, 1) for a, b in combinations(means, 2)]
        candidates = distances[index, np.isin(masks[index], [3, 5, 6])]
        if not len(candidates):
            continue
        for name, distance in (("all_distinct_occurrences", distances[index, index]),
                               ("different_characters", np.mean(pair_distances))):
            delta = candidates - distance
            count = sum(map(len, groups_for_label.values()))
            rows.append({"system": system, "pinyin": label, "estimator": name, "occurrences": count,
                         "support": "2–10" if count <= 10 else ">10",
                         "characters": len(means), "character_pairs": len(pair_distances),
                         "candidates": len(candidates), "distance": distance,
                         "wins": (delta > tolerance).sum(), "losses": (delta < -tolerance).sum(),
                         "ties": (abs(delta) <= tolerance).sum()})
    return rows


def reference_b_matrix(labels):
    from ..evaluation.dimsim_pypinyin import PypinyinDimsimDistance

    metric = PypinyinDimsimDistance()
    matrix = np.zeros((len(labels), len(labels)))
    for i, j in combinations(range(len(labels)), 2):
        value = metric(labels[i], labels[j])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid DIMSIM B distance for {labels[i]} and {labels[j]}")
        matrix[i, j] = matrix[j, i] = value
    return matrix, metric.provenance


def geometry_tables(labels, components, reference, distances, systems, tolerance):
    masks = component_masks(components)
    baseline = sum((masks & bit) == 0 for bit in (4, 2, 1)).astype(float)
    missing = np.flatnonzero(~np.isfinite(distances[0].diagonal()))
    baseline[missing, missing] = np.nan
    return evaluate_geometry(labels, components, np.stack([reference, *distances, baseline]),
                             systems=systems, tolerance=tolerance, contextual=True)[0]


TABLE_FIELDS = {
    "support": "seed pinyin excluded_from_primary occurrences characters",
    "support_bins": "seed occurrence_bin character_bin syllables",
    "coverage": "seed population syllables occurrences full_matches",
    "metrics": "seed population system metric slice value comparisons reference_ties model_ties",
    "paired": "seed population from_system to_system metric slice difference",
    "full_match": "seed system pinyin estimator occurrences support characters character_pairs candidates distance wins losses ties",
    "reference_agreement": "seed slice system triples dimsim_a_ties dimsim_b_ties comparisons model_ties agreement_a agreement_b",
    "residuals": "seed population system rank left right mask residual",
}


def add_seed_tables(rows, seed, populations, matrices, reference_a, reference_b, components,
                    systems, tolerance, primary_tables, paired_differences):
    evaluations = {"primary": primary_tables}
    for name, (inventory, indices, positions, match) in populations.items():
        rows["coverage"].append({"seed": seed, "population": name, "syllables": len(inventory),
                                 "occurrences": len(positions[seed]), "full_matches": len(match)})
        if len(inventory) < 2:
            continue
        reference = reference_a[np.ix_(indices, indices)]
        component_rows = [components[label] for label in inventory]
        evaluations[name] = geometry_tables(inventory, component_rows, reference, matrices[name], systems, tolerance)
        if name == "expanded":
            rows["reference_agreement"].extend(
                {"seed": seed, **row} for row in strict_reference_agreement(
                    component_rows, reference, reference_b, matrices[name], systems[1:-1], tolerance
                )
            )
    for name, tables in evaluations.items():
        for row in tables["pairwise_spearman"]:
            rows["metrics"].append({"seed": seed, "population": name, "system": row["system"],
                                    "metric": "Spearman", "slice": "Distinct syllables",
                                    "value": row["spearman"], "comparisons": row["pairs"]})
        for table, metric, dimension in (
            ("dimsim_agreement", "DIMSIM Agreement", "slice"),
            ("same_mask_agreement", "Same mask agreement", "mask"),
            ("controlled_match", "Controlled match win rate", "from_mask"),
        ):
            for row in tables[table]:
                if table == "controlled_match" and row[dimension] != "All":
                    continue
                rows["metrics"].append({
                    "seed": seed, "population": name, "system": row["system"], "metric": metric,
                    "slice": row[dimension], "value": row.get("agreement", row.get("win_rate")),
                    "comparisons": row.get("comparisons", row["triples"]),
                    "reference_ties": row.get("dimsim_ties"), "model_ties": row["model_ties"],
                })
        rows["paired"].extend({"seed": seed, "population": name, **row} for row in paired_differences(tables))
        if name in ("primary", "unk_retained"):
            rows["residuals"].extend({"seed": seed, "population": name, **row}
                                     for row in tables["residuals_top50"])
