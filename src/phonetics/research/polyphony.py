"""Exact occurrence-level polyphony scores with hierarchical, frequency-safe weights."""
from collections import defaultdict
from itertools import combinations

import numpy as np

METRICS = ("score", "delta", "within_distance", "cross_distance", "win_rate", "tie_rate", "loss_rate")
SLICES = {"main": 2, "high_support": 11}
POLICY = {
    "support": SLICES, "control_support": "same threshold as target readings",
    "unknown_policy": "exclude target UNK", "distance": "cosine",
    "weighting": "equal characters, pairs, directions; equal triples within direction",
    "control_weighting": "equal control characters, equal occurrences within each character",
    "score": "1 if cross-within > tolerance; 0.5 on ties; 0 otherwise",
    "control_gap": "matched main minus control; score gap in percentage points",
}


def plan_polyphony(occurrences, indices_by_seed):
    groups = {}
    for seed, indices in indices_by_seed.items():
        groups[seed] = defaultdict(list)
        for i in indices:
            row = occurrences[i]
            groups[seed][row["character"], row["pinyin"]].append(i)
    observed = defaultdict(set)
    for row in occurrences:
        observed[row["character"]].add(row["pinyin"])
    candidates = [(char, a, b) for char, readings in sorted(observed.items())
                  for a, b in combinations(sorted(readings), 2)]
    plan, coverage = [], []
    for name, minimum in SLICES.items():
        for char, a, b in candidates:
            eligible = all(len(g.get((char, a), ())) >= minimum and len(g.get((char, b), ())) >= minimum
                           for g in groups.values())
            controls = {}
            for reading in (a, b):
                controls[reading] = sorted(y for y in observed if y != char and
                    all(len(g.get((y, reading), ())) >= minimum for g in groups.values())) if eligible else []
            matched = eligible and all(controls.values())
            for seed, g in groups.items():
                coverage.append(dict(slice=name, seed=seed, character=char, reading_a=a, reading_b=b,
                    count_a=len(g.get((char, a), ())), count_b=len(g.get((char, b), ())), eligible=eligible,
                    matched_control=matched, controls_a=len(controls[a]), controls_b=len(controls[b]),
                    reason="included" if eligible else "insufficient support in at least one seed"))
            if eligible:
                plan.append(dict(slice=name, character=char, reading_a=a, reading_b=b,
                                 controls=controls, matched_control=matched))
    return groups, plan, coverage


def direction_score(anchor, other, tolerance):
    """Count all i!=j,k comparisons without constructing the cubic triple array."""
    return _directions(anchor, [other], tolerance)[0]


def _directions(anchor, candidates, tolerance):
    totals = np.zeros((len(candidates), len(METRICS)), dtype=np.float64)
    n = len(anchor)
    if n < 2 or any(not len(other) for other in candidates):
        raise ValueError("a direction requires two anchors and at least one candidate")
    # Bound peak memory even for a frequent character. Input vectors are normalized.
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        within = np.clip(1 - anchor[start:stop] @ anchor.T, 0, 2)
        for group, other in enumerate(candidates):
            cross = np.clip(1 - anchor[start:stop] @ other.T, 0, 2)
            for offset, i in enumerate(range(start, stop)):
                w = np.delete(within[offset], i)
                c = np.sort(cross[offset])
                low = np.searchsorted(c, w - tolerance, side="left")
                high = np.searchsorted(c, w + tolerance, side="right")
                score = np.mean((len(c) - high + .5 * (high - low)) / len(c))
                wm, cm = float(w.mean()), float(c.mean())
                win, tie, loss = (float(np.mean(x / len(c))) for x in (len(c)-high, high-low, low))
                totals[group] += (score, cm - wm, wm, cm, win, tie, loss)
    return [dict(zip(METRICS, map(float, row / n))) for row in totals]


def _average(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def evaluate_polyphony(hidden, groups, plan, tolerance, system):
    hidden = np.asarray(hidden, dtype=np.float64)
    hidden = hidden / np.linalg.norm(hidden, axis=1, keepdims=True)
    directed, pairs, characters, summary = [], [], [], []
    for item in plan:
        char, a, b = (item[k] for k in ("character", "reading_a", "reading_b"))
        values = defaultdict(list)
        for source, target in ((a, b), (b, a)):
            anchor = hidden[groups[char, source]]
            controls = item["controls"][target] if item["matched_control"] else []
            results = _directions(anchor, [hidden[groups[y, target]] for y in (char, *controls)], tolerance)
            main = results[0]
            base = dict(slice=item["slice"], system=system, character=char,
                        reading_a=a, reading_b=b, direction=f"{source} -> {target}")
            def record(pop, result, candidates, controls, raw_results):
                multiplier = len(anchor)*(len(anchor)-1)
                return {**base, "population": pop, **result, "anchors": len(anchor),
                    "candidates": candidates, "triples": multiplier*candidates,
                    "wins": int(round(sum(r["win_rate"]*multiplier*n for r,n in raw_results))),
                    "ties": int(round(sum(r["tie_rate"]*multiplier*n for r,n in raw_results))),
                    "losses": int(round(sum(r["loss_rate"]*multiplier*n for r,n in raw_results))),
                    "control_characters": controls}
            main_count = len(groups[char, target])
            directed.append(record("main", main, main_count, 0, [(main, main_count)]))
            values["main"].append(main)
            if item["matched_control"]:
                control_parts = [(r, len(groups[y,target])) for r,y in zip(results[1:], controls)]
                control = _average([r for r,n in control_parts])
                for pop, value in (("matched_main", main), ("control", control)):
                    raw = [(main, main_count)] if pop == "matched_main" else control_parts
                    directed.append(record(pop, value, sum(n for r,n in raw), len(control_parts), raw))
                    values[pop].append(value)
        for pop, rows in values.items():
            support = [r for r in directed if r["slice"] == item["slice"] and r["character"] == char
                       and r["reading_a"] == a and r["reading_b"] == b and r["population"] == pop]
            pairs.append(dict(slice=item["slice"], system=system, character=char, reading_a=a,
                              reading_b=b, population=pop,
                              **{k: sum(r[k] for r in support) for k in ("triples", "wins", "ties", "losses")},
                              **_average(rows)))
    for name in SLICES:
        for pop in ("main", "matched_main", "control"):
            by_char = defaultdict(list)
            for row in pairs:
                if row["slice"] == name and row["population"] == pop:
                    by_char[row["character"]].append(row)
            for char, rows in sorted(by_char.items()):
                characters.append(dict(slice=name, system=system, population=pop, character=char,
                                       pairs=len(rows),
                                       **{k: sum(r[k] for r in rows) for k in ("triples", "wins", "ties", "losses")},
                                       **_average(rows)))
            rows = [r for r in characters if r["slice"] == name and r["population"] == pop]
            summary.append(dict(slice=name, system=system, population=pop, characters=len(rows),
                pairs=sum(r["pairs"] for r in rows), **(_average(rows) if rows else dict.fromkeys(METRICS))))
        matched, control = summary[-2:]
        summary.append(dict(slice=name, system=system, population="main_minus_control",
            characters=matched["characters"], pairs=matched["pairs"],
            **{k: ((matched[k]-control[k]) * (100 if k == "score" else 1)
                   if matched[k] is not None else None) for k in METRICS}))
    return dict(polyphony_directions=directed, polyphony_pairs=pairs,
                polyphony_characters=characters, polyphony_summary=summary)
