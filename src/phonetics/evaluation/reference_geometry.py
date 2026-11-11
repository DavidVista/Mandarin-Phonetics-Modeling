"""Syllable distance views and component comparisons for reference geometry."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

SYSTEMS = ("DIMSIM", "Wav2vec2 base", "Wav2vec2 zh-cn")
SLICES = ("Controlled match", "Trade-off", "Same mask")
TRANSITIONS = tuple((low, high) for low in range(7) for high in range(low + 1, 8)
                    if low & high == low)
MASKS = tuple(f"{mask:03b}" for mask in range(8))


def component_masks(components):
    components = np.asarray(components)
    if components.ndim != 2 or components.shape[1] != 3:
        raise ValueError("expected initial, final and tone for each label")
    return sum(weight * (components[:, None, k] == components[None, :, k])
               for k, weight in enumerate((4, 2, 1))).astype(np.uint8)


def _sign(values, tolerance):
    return (values > tolerance).astype(np.int8) - (values < -tolerance)


def _rate(score, count):
    return float(score / count) if count else None


def evaluate_geometry(labels, components, distances, *, tolerance, systems=SYSTEMS, contextual=False):
    """Process one anchor at a time. Map projections never drive table weights."""
    n = len(labels)
    distances = np.asarray(distances, dtype=np.float64)
    count_systems = len(systems)
    if distances.shape != (count_systems, n, n):
        raise ValueError("distance matrices do not match the system and label inventories")
    diagonal = distances.diagonal(axis1=1, axis2=2)
    availability = np.isfinite(diagonal[1:]).all(axis=0)
    off_diagonal = distances[:, ~np.eye(n, dtype=bool)]
    if (n < 2 or len(set(labels)) != n or len(components) != n or not np.isfinite(off_diagonal).all()
            or np.any(distances < 0) or np.isinf(distances).any()
            or not np.array_equal(distances, distances.transpose(0, 2, 1), equal_nan=True)
            or np.any(diagonal[0]) or (not contextual and np.any(diagonal))
            or (contextual and not np.all(np.isfinite(diagonal[1:]) == availability))):
        raise ValueError("invalid shared reference geometry distance views")
    masks = component_masks(components)
    if np.count_nonzero(masks == 7) != n:
        raise ValueError("duplicate component triples")
    left, right = np.triu_indices(n, k=1)
    candidate_changes = 7 ^ masks[left, right]
    planned = np.zeros(len(TRANSITIONS), dtype=np.int64)
    transition_totals = np.zeros((len(TRANSITIONS), count_systems, 5))
    agreement_totals = np.zeros((3, count_systems, 3))
    controlled_anchors = np.zeros((len(TRANSITIONS), n), dtype=bool)
    agreement_anchors = np.zeros((3, n), dtype=bool)
    anchor_wins = np.zeros((n, count_systems))
    anchor_support = np.zeros(n, dtype=np.int64)
    projections = np.full((2, count_systems, n, n), np.nan)
    projection_counts = np.zeros((2, n, n), dtype=np.int32)
    records, controlled_values = [], []
    transition_index = np.full((8, 8), -1)
    for index, (low, high) in enumerate(TRANSITIONS):
        transition_index[low, high] = index

    for anchor in range(n):
        ml, mr = masks[anchor, left], masks[anchor, right]
        low, high = np.minimum(ml, mr), np.maximum(ml, mr)
        shared = low & high
        transition = transition_index[low, high]
        controlled = (transition >= 0) & (candidate_changes == (low ^ high))
        distinct = (left != anchor) & (right != anchor)
        trade = distinct & (shared != low)
        same = distinct & (low == high)
        anchor_distances = distances[:, anchor]
        planned += np.bincount(transition[controlled], minlength=len(TRANSITIONS))
        if not availability[anchor]:
            controlled &= distinct
        chosen = np.flatnonzero(controlled)
        if chosen.size:
            b = np.where(ml[chosen] > mr[chosen], left[chosen], right[chosen])
            c = np.where(ml[chosen] > mr[chosen], right[chosen], left[chosen])
            codes = transition[chosen]
            effects = anchor_distances[:, c] - anchor_distances[:, b]
            signs = _sign(effects, tolerance)
            ref_valid = signs[0] != 0
            wins = (signs + 1) / 2
            scores = (1 + signs * signs[0]) / 2
            anchor_wins[anchor] = wins.sum(axis=1)
            anchor_support[anchor] = len(b)
            agreement_anchors[0, anchor] = ref_valid.any()
            agreement_totals[0, :, 0] += len(b)
            agreement_totals[0, :, 1] += ref_valid.sum()
            agreement_totals[0, :, 2] += scores[:, ref_valid].sum(axis=1)
            for code in np.unique(codes):
                select = codes == code
                candidates, inverse = np.unique(b[select], return_inverse=True)
                count = np.bincount(inverse)
                valid = ref_valid[select]
                agree_count = np.bincount(inverse[valid], minlength=len(count))
                controlled_anchors[code, anchor] = True
                rows = []
                for model in range(count_systems):
                    effect, win, score = effects[model, select], wins[model, select], scores[model, select]
                    sums = np.bincount(inverse, weights=effect)
                    successes = np.bincount(inverse, weights=win)
                    agreements = np.bincount(inverse[valid], weights=score[valid], minlength=len(count))
                    transition_totals[code, model] += [len(effect), win.sum(), effect.sum(), valid.sum(), score[valid].sum()]
                    agreement = np.divide(agreements, agree_count, out=np.full(len(count), np.nan), where=agree_count > 0)
                    rows.append(np.column_stack((sums / count, successes / count, agreement)))
                records.extend(zip(np.full(len(count), code), np.full(len(count), anchor), candidates, count, agree_count))
                controlled_values.append(np.stack(rows, axis=1))
        for index, selection in enumerate((trade, same), 1):
            a, b = left[selection], right[selection]
            signs = _sign(anchor_distances[:, b] - anchor_distances[:, a], tolerance)
            valid = signs[0] != 0
            scores = (1 + signs[:, valid] * signs[0, valid]) / 2
            agreement_anchors[index, anchor] = valid.any()
            agreement_totals[index, :, 0] += len(a)
            agreement_totals[index, :, 1] += valid.sum()
            agreement_totals[index, :, 2] += scores.sum(axis=1)
            endpoints = np.r_[a[valid], b[valid]]
            counts = np.bincount(endpoints, minlength=n)
            projection_counts[index - 1, anchor] = counts
            for model in range(count_systems):
                totals = np.bincount(endpoints, weights=np.tile(scores[model], 2), minlength=n)
                np.divide(totals, counts, out=projections[index - 1, model, anchor], where=counts > 0)

    controlled_rows, agreement_rows, anchor_rows, anchor_summary, rank_rows = [], [], [], [], []
    for model, name in enumerate(systems):
        for index, (low, high) in enumerate((*TRANSITIONS, (None, None))):
            totals = transition_totals[:, model].sum(axis=0) if low is None else transition_totals[index, model]
            count, wins, effect, ref_count, agree = totals
            anchors = controlled_anchors.any(axis=0) if low is None else controlled_anchors[index]
            controlled_rows.append({
                **({"planned_triples": int(planned.sum() if low is None else planned[index])} if contextual else {}),
                "system": name, "from_mask": MASKS[low] if low is not None else "All",
                "to_mask": MASKS[high] if high is not None else "All", "triples": int(count),
                "anchors": int(anchors.sum()), "win_rate": _rate(wins, count),
                "mean_effect": _rate(effect, count), "dimsim_agreement": _rate(agree, ref_count),
                "dimsim_comparisons": int(ref_count),
            })
        for index, name_slice in enumerate(SLICES):
            count, valid, score = agreement_totals[index, model]
            agreement_rows.append({"system": name, "slice": name_slice, "triples": int(count),
                                   "anchors": int(agreement_anchors[index].sum()), "dimsim_ties": int(count - valid),
                                   "comparisons": int(valid), "agreement": _rate(score, valid)})
        for anchor, label in enumerate(labels):
            anchor_rows.append({"system": name, "anchor": label, "triples": int(anchor_support[anchor]),
                                "win_rate": _rate(anchor_wins[anchor, model], anchor_support[anchor])})
        supported = anchor_support > 0
        rates = anchor_wins[supported, model] / anchor_support[supported]
        quartiles = np.quantile(rates, (.25, .5, .75)).tolist() if len(rates) else [None] * 3
        anchor_summary.append({"system": name, "anchors": len(rates),
                               "q25": quartiles[0], "median": quartiles[1], "q75": quartiles[2]})
        reference, values = distances[0, left, right], distances[model, left, right]
        rho = float(spearmanr(reference, values).statistic) if np.ptp(reference) and np.ptp(values) else None
        rank_rows.append({"system": name, "pairs": len(left), "spearman": rho})

    tables = {"controlled_match": controlled_rows, "dimsim_agreement": agreement_rows,
              "anchor_win_rates": anchor_rows, "anchor_summary": anchor_summary,
              "pairwise_spearman": rank_rows}
    tables.update(_profiles(labels, distances, masks, left, right, systems=systems))
    maps = {
        "masks": masks, "controlled_cells": np.asarray(records, dtype=np.int32).reshape(-1, 5),
        "controlled_values": np.concatenate(controlled_values) if controlled_values else np.empty((0, count_systems, 3)),
        "agreement": projections, "agreement_counts": projection_counts,
        "neighborhood": np.argsort(distances[0], axis=1, kind="stable"),
        "component_order": np.asarray(sorted(range(n), key=lambda index: (tuple(components[index]), labels[index]))),
    }
    return tables, maps


def _profiles(labels, distances, masks, left, right, *, systems):
    n = len(labels)
    rows, summary, gains, residuals = [], [], [], []
    for model, system in enumerate(systems):
        similarity = -distances[model] if model == 0 else 1 - distances[model]
        values, pair_masks = similarity[left, right], masks[left, right]
        counts = np.bincount(pair_masks, minlength=8)
        sums = np.bincount(pair_masks, weights=values, minlength=8)
        diagonal = similarity.diagonal()
        diagonal = diagonal[np.isfinite(diagonal)]
        counts[7] = len(diagonal)
        sums[7] = diagonal.sum()
        means = np.divide(sums, counts, out=np.full(8, np.nan), where=counts > 0)
        for mask in range(8):
            rows.append({"system": system, "mask": MASKS[mask], "pairs": int(counts[mask]),
                         "mean_similarity": float(means[mask]) if counts[mask] else None})
        error = values - means[pair_masks]
        for include_diagonal in (False, True):
            y = np.r_[values, diagonal] if include_diagonal else values
            sse = np.dot(error, error)
            if include_diagonal and len(diagonal):
                diagonal_error = diagonal - means[7]
                sse += np.dot(diagonal_error, diagonal_error)
            sst = np.dot(y - y.mean(), y - y.mean())
            summary.append({"system": system, "includes_full_match": include_diagonal,
                            "pairs": len(y), "r_squared": float(1 - sse / sst) if sst else None})
        for component, bit in zip(("Initial", "Final", "Tone"), (4, 2, 1)):
            ordinary = [mask for mask in range(7) if not mask & bit and mask | bit != 7]
            steps = [float(means[mask | bit] - means[mask]) for mask in ordinary]
            full = float(means[7] - means[7 ^ bit])
            average = float(np.mean(steps))
            row = {"system": system, "component": component}
            for index, (mask, value) in enumerate(zip(ordinary, steps), 1):
                row[f"step_{index}"] = f"{MASKS[mask]} to {MASKS[mask | bit]}"
                row[f"gain_{index}"] = value if np.isfinite(value) else None
            row.update(ordinary_mean=average if np.isfinite(average) else None,
                       full_match_gain=full if np.isfinite(full) else None,
                       j=full - average if np.isfinite(full - average) else None)
            gains.append(row)
        # Inputs have lexical labels, so stable sorting also fixes residual ties.
        for rank, index in enumerate(np.argsort(-error, kind="stable")[:50], 1):
            if error[index] <= 0:
                break
            residuals.append({"system": system, "rank": rank, "left": labels[left[index]],
                              "right": labels[right[index]], "mask": MASKS[pair_masks[index]],
                              "residual": float(error[index])})
    return {"mask_profiles": rows, "mask_summary": summary, "full_match_gains": gains,
            "residuals_top50": residuals}
