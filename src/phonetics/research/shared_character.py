"""Shared-character comparisons on the seed-specific valid reading inventory."""
from collections import Counter, defaultdict

import numpy as np


SLICES = ("any_shared_character", "shared_character_at_least_25pct")


def _rate(value, count):
    return float(value / count) if count else None


def reading_counts(labels, occurrences):
    counts = {label: Counter() for label in labels}
    for row in occurrences:
        counts[row["pinyin"]][row["character"]] += 1
    return list(counts.values())


def _shared_readings(counts, eligible, frequency):
    by_character = defaultdict(set)
    for index in eligible:
        for character in counts[index]:
            by_character[character].add(index)
    for anchor in eligible:
        any_shared = set().union(*(by_character[char] for char in counts[anchor])) - {anchor}
        strong_shared = {other for char, count in counts[anchor].items() if 4 * count >= frequency[anchor]
                         for other in by_character[char] if other != anchor
                         and 4 * counts[other][char] >= frequency[other]}
        yield anchor, any_shared, strong_shared


def shared_pair_residuals(counts, masks, distances, systems, eligible):
    frequency = np.asarray([sum(group.values()) for group in counts])
    shared = {a: (any_shared, strong) for a, any_shared, strong in _shared_readings(counts, eligible, frequency)}
    left, right = np.triu_indices(len(counts), 1)
    pair_masks = masks[left, right]
    similarity = 1 - distances[:, left, right]
    similarity[0] = -distances[0, left, right]
    residual = similarity.copy()
    for code in range(7):
        selected = pair_masks == code
        if selected.any():
            residual[:, selected] -= similarity[:, selected].mean(axis=1, keepdims=True)
    strata = defaultdict(lambda: [[], [], []])
    for index in np.flatnonzero(np.isin(left, eligible) & np.isin(right, eligible)):
        a, b = left[index], right[index]
        any_shared, strong = shared[a]
        bins = sorted((int(frequency[a]).bit_length() - 1, int(frequency[b]).bit_length() - 1))
        groups = strata[int(pair_masks[index]), *bins]
        if b in any_shared:
            groups[1].append(index)
            if b in strong:
                groups[2].append(index)
        else:
            groups[0].append(index)
    for slice_index, slice_name in enumerate(SLICES, 1):
        sums = np.zeros((8, 2, len(systems)))
        support = np.zeros((8, 4), dtype=int)
        for (code, *_), groups in strata.items():
            controls, exposed = groups[0], groups[slice_index]
            for mask_index in (code, 7):
                support[mask_index, 0] += len(exposed)
                if controls and exposed:
                    sums[mask_index, 0] += residual[:, exposed].sum(axis=1)
                    sums[mask_index, 1] += len(exposed) * residual[:, controls].mean(axis=1)
                    support[mask_index, 1:] += [len(exposed), len(controls), 1]
        for code in (7, *range(7)):
            total, matched, controls, strata_count = map(int, support[code])
            for index, system in enumerate(systems):
                shared_mean, control_mean = (_rate(value, matched) for value in sums[code, :, index])
                yield dict(slice=slice_name, system=system, mask="all" if code == 7 else f"{code:03b}",
                           delta=shared_mean - control_mean if matched else None,
                           mean_shared=shared_mean, mean_control_weighted=control_mean,
                           shared_pairs_total=total, shared_pairs_matched=matched,
                           control_pairs_matched=controls, matched_strata=strata_count)
