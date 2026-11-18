"""Shared-character comparisons on the seed-specific valid reading inventory."""
from collections import Counter, defaultdict

import numpy as np

SLICES = ("any_shared_character", "shared_character_at_least_25pct")
GROUPS = ("Controlled match", "Trade-off", "Same mask", "Same mask 000", "Same mask excluding 000")
TRIPLE_BANDS = (20, 40, 80, 160, 320, 640, 1280, 2560)


def same_mask_breakdown(rows):
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    for system, entries in by_system.items():
        zero = [row for row in entries if row["mask"] == "000"]
        other = [row for row in entries if row["mask"] != "000"]
        for scope, selected in zip(GROUPS[2:], (entries, zero, other), strict=True):
            totals = {key: sum(row[key] for row in selected) for key in
                      ("triples", "comparisons", "wins", "losses", "model_ties", "dimsim_ties")}
            yield dict(system=system, scope=scope, **totals,
                       agreement=_rate(totals["wins"] + .5 * totals["model_ties"], totals["comparisons"]))


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


def _classes(masks, anchor, left, right):
    ml, mr = masks[anchor, left], masks[anchor, right]
    same = ml == mr
    trade = (ml & mr) != np.minimum(ml, mr)
    controlled = (~same & ~trade & ((7 ^ masks[left, right]) == (ml ^ mr)))
    return np.select((controlled, trade, same), (0, 1, 2), default=-1)


def _scores(distances, tolerance, anchor, left, right):
    gap = distances[:, anchor, right] - distances[:, anchor, left]
    sign = (gap > tolerance).astype(np.int8) - (gap < -tolerance).astype(np.int8)
    ordered = sign[0] != 0
    return int(ordered.sum()), ((1 + sign[:, ordered] * sign[0, ordered]) / 2).sum(axis=1)


def _control_pairs(pool, left_key, right_key):
    left, right = pool.get(left_key, ()), pool.get(right_key, ())
    if left_key == right_key:
        left = np.asarray(left, dtype=np.intp)
        i, j = np.triu_indices(len(left), 1)
        return left[i], left[j]
    return np.repeat(left, len(right)).astype(np.intp), np.tile(right, len(left)).astype(np.intp)


def shared_triple_agreement(counts, masks, distances, systems, tolerance):
    frequency = np.asarray([sum(group.values()) for group in counts])
    eligible = np.flatnonzero(frequency >= 11)
    bands = np.searchsorted(TRIPLE_BANDS, frequency, side="right")
    totals = np.zeros((2, 5, 5), dtype=int)
    credits = np.zeros((2, 5, 3, len(systems)))
    matched_pairs = defaultdict(set)
    matched_anchors = defaultdict(set)
    for anchor, shared, strong in _shared_readings(counts, eligible, frequency):
        if not shared:
            continue
        pool = defaultdict(list)
        for other in eligible:
            if other != anchor and other not in shared:
                pool[int(masks[anchor, other]), int(bands[other])].append(other)
        pool = {key: np.asarray(value, dtype=np.intp) for key, value in pool.items()}
        exposed = defaultdict(lambda: ([], []))
        for candidate in sorted(shared):
            left_key = (int(masks[anchor, candidate]), int(bands[candidate]))
            for right_key, controls in pool.items():
                classes = _classes(masks, anchor, candidate, controls)
                for kind in np.unique(classes):
                    if kind < 0:
                        continue
                    selected = controls[classes == kind]
                    for slice_index in range(2 if candidate in strong else 1):
                        lefts, rights = exposed[slice_index, kind, left_key, right_key]
                        lefts.extend([candidate] * len(selected))
                        rights.extend(selected)
        cache = {}
        for (s, kind, left_key, right_key), (lefts, rights) in exposed.items():
            count, score = _scores(distances, tolerance, anchor, np.asarray(lefts), np.asarray(rights))
            if not count:
                continue
            key = (kind, *sorted((left_key, right_key)))
            if key not in cache:
                left, right = _control_pairs(pool, left_key, right_key)
                selected = _classes(masks, anchor, left, right) == kind
                cache[key] = _scores(distances, tolerance, anchor, left[selected], right[selected])
            control_count, control_score = cache[key]
            groups = (kind,) if kind != 2 else (kind, 3 if left_key[0] == 0 else 4)
            for group in groups:
                totals[s, group] += [count, count if control_count else 0, bool(control_count),
                                     not control_count, control_count]
                credits[s, group, 0] += score
                if control_count:
                    credits[s, group, 1] += score
                    credits[s, group, 2] += count / control_count * control_score
                    left, right = np.asarray(lefts), np.asarray(rights)
                    ordered = np.abs(distances[0, anchor, right] - distances[0, anchor, left]) > tolerance
                    support = np.union1d(left[ordered], right[ordered])
                    matched_pairs[s, group].update(
                        (min(anchor, other), max(anchor, other)) for other in support
                    )
                    matched_anchors[s, group].add(anchor)
    for s, slice_name in enumerate(SLICES):
        for g, comparison in enumerate(GROUPS):
            count, matched, matched_strata, unmatched_strata, controls = map(int, totals[s, g])
            for i, system in enumerate(systems):
                all_score, shared, control = credits[s, g, :, i]
                yield dict(slice=slice_name, comparison=comparison, system=system,
                           shared_char_comparisons=count, matched_shared_char_comparisons=matched,
                           matched_share=_rate(matched, count), matched_strata=matched_strata,
                           matched_shared_char_pairs=len(matched_pairs[s, g]),
                           matched_anchors=len(matched_anchors[s, g]),
                           unmatched_strata=unmatched_strata, control_comparisons_in_matched_strata=controls,
                           shared_char_agreement_all=_rate(all_score, count),
                           shared_char_agreement_matched=_rate(shared, matched),
                           control_agreement_weighted=_rate(control, matched),
                           delta_percentage_points=_rate(100 * (shared - control), matched))
