"""Explicit, non-transitive phonetic-neighbour annotations for Mandarin pairs.

The relations in this module are an explanatory expert whitelist based on small
articulatory contrasts.  They are not a perceptually validated threshold and
are deliberately independent of the DIMSIM distance.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


def _pairs(values: Iterable[tuple[str, str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(pair) for pair in values)


INITIAL_RELATIONS = {
    "aspiration": _pairs((("b", "p"), ("d", "t"), ("g", "k"), ("j", "q"), ("z", "c"), ("zh", "ch"))),
    "sibilant_place": _pairs((("z", "zh"), ("c", "ch"), ("s", "sh"), ("j", "zh"), ("q", "ch"),
                               ("x", "sh"), ("j", "z"), ("q", "c"), ("x", "s"))),
    "manner": _pairs((("z", "s"), ("c", "s"), ("zh", "sh"), ("ch", "sh"), ("j", "x"), ("q", "x"))),
}
INITIAL_SENSITIVITY_RELATIONS = {
    "dialect_learner_looser": _pairs((("n", "l"), ("l", "r"), ("sh", "r"), ("f", "h"), ("h", "x"))),
}
FINAL_RELATIONS = {
    "nasal_coda": _pairs((("an", "ang"), ("en", "eng"), ("in", "ing"), ("uan", "uang"))),
    # These involve real frontness/rounding changes; "near" does not mean identical.
    "frontness_rounding": _pairs((("i", "ü"), ("ie", "üe"), ("ian", "üan"), ("in", "ün"))),
    "apical_place": _pairs((("apical_dental", "apical_postalveolar"),)),
}
FINAL_EXTENDED_RELATIONS = {
    "glide_or_rime_extension": _pairs(
        (("a", "ia"), ("a", "ua"), ("ai", "uai"), ("an", "ian"), ("an", "uan"),
         ("ang", "iang"), ("ang", "uang"), ("ao", "iao"), ("e", "ie"), ("e", "üe"),
         ("en", "in"), ("en", "uen"), ("en", "ün"), ("eng", "ing"), ("eng", "ong"),
         ("i", "ia"), ("i", "ie"), ("in", "ian"), ("ing", "iang"), ("ing", "iong"),
         ("o", "uo"), ("ou", "iou"), ("u", "ua"), ("u", "uo"), ("uan", "üan"),
         ("uen", "ün"), ("u", "ü"))
    ),
}

def _inventory_rows(component: str, profile: str, groups: dict[str, frozenset[frozenset[str]]], reason: str):
    return [
        {"component": component, "profile": profile, "family": family,
         "pair": "–".join(sorted(pair)), "reason": reason}
        for family, relations in groups.items() for pair in sorted(relations, key=lambda p: sorted(p))
    ]


COMPONENT_INVENTORY = (
    _inventory_rows("initial", "core", INITIAL_RELATIONS, "Small explicit articulatory contrast.")
    + _inventory_rows("initial", "extended", INITIAL_SENSITIVITY_RELATIONS,
                      "Dialect, learner, or deliberately looser sensitivity relation.")
    + _inventory_rows("final", "core", FINAL_RELATIONS,
                      "Explicit nasal, rounding/frontness, or apical-place contrast; real vowel effects remain.")
    + _inventory_rows("final", "extended", FINAL_EXTENDED_RELATIONS,
                      "Broader glide/rime sensitivity relation; u–ü is extension-only.")
)

_ALIASES = {"ui": "uei", "iu": "iou", "un": "uen", "v": "ü", "van": "üan", "ve": "üe", "vn": "ün"}
_DENTAL = frozenset(("z", "c", "s"))
_POSTALVEOLAR = frozenset(("zh", "ch", "sh", "r"))
_PALATAL = frozenset(("j", "q", "x"))


def normalize_final(initial: Any, final: Any) -> str:
    """Return a robust canonical final, preserving contextual apical ``i``."""
    ini = str(initial).strip().lower()
    raw = str(final).strip().lower().replace("u:", "ü")
    if raw == "i" and ini in _DENTAL:
        return "apical_dental"
    if raw == "i" and ini in _POSTALVEOLAR:
        return "apical_postalveolar"
    canonical = _ALIASES.get(raw, raw)
    if ini in _PALATAL:
        canonical = {"u": "ü", "uan": "üan", "uen": "ün", "ue": "üe"}.get(canonical, canonical)
    return canonical


def _relation(a: str, b: str, relations: dict[str, frozenset[frozenset[str]]]) -> str:
    if a == b:
        return ""
    target = frozenset((a, b))
    return next((category for category, pairs in relations.items() if target in pairs), "")


def classify_pairs(columns: dict[str, np.ndarray], profile: str = "core") -> dict[str, np.ndarray]:
    """Classify array-oriented pair columns without transitive expansion."""
    if profile not in {"core", "extended"}:
        raise ValueError("profile must be 'core' or 'extended'")
    required = ("left_gold_initial", "right_gold_initial", "left_gold_final", "right_gold_final")
    missing = [key for key in required if key not in columns]
    if missing:
        raise KeyError(f"missing pair columns: {', '.join(missing)}")
    n = len(columns[required[0]])
    if any(len(columns[key]) != n for key in required):
        raise ValueError("pair columns must have equal lengths")
    li = np.asarray(columns["left_gold_initial"]).astype(str)
    ri = np.asarray(columns["right_gold_initial"]).astype(str)
    lf = np.asarray([normalize_final(a, b) for a, b in zip(li, columns["left_gold_final"])])
    rf = np.asarray([normalize_final(a, b) for a, b in zip(ri, columns["right_gold_final"])])
    initial_relations = dict(INITIAL_RELATIONS)
    final_relations = dict(FINAL_RELATIONS)
    if profile == "extended":
        initial_relations.update(INITIAL_SENSITIVITY_RELATIONS)
        final_relations.update(FINAL_EXTENDED_RELATIONS)
    relation_initial = np.asarray([_relation(a, b, initial_relations) for a, b in zip(li, ri)])
    relation_final = np.asarray([_relation(a, b, final_relations) for a, b in zip(lf, rf)])
    close_initial = relation_initial != ""
    close_final = relation_final != ""
    equal_final = lf == rf
    both = close_initial & close_final
    recognition = np.select(
        (both, close_initial, close_final), ("both", "initial_only", "final_only"), default="neither"
    )
    return {
        "left_canonical_final": lf, "right_canonical_final": rf,
        "close_initial": close_initial, "close_final": close_final,
        "normalized_equal_final": equal_final,
        "relation_initial": relation_initial, "relation_final": relation_final,
        "recognition": recognition, "exclusive_class": recognition.copy(),
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def summarize(columns: dict[str, np.ndarray], profile: str = "core") -> dict[str, Any]:
    """Summarize whitelist prevalence in fixed DIMSIM bins, focused on mask 000."""
    classified = classify_pairs(columns, profile)
    raw_distance = np.asarray(columns["distance"])
    if raw_distance.dtype.kind not in "biuf":
        raise TypeError("distance must be a numeric array")
    distance = raw_distance.astype(float, copy=False)
    if len(distance) != len(classified["close_initial"]):
        raise ValueError("distance and component columns must have equal lengths")
    if not np.isfinite(distance).all():
        raise ValueError("distance must be strictly numeric and finite")
    if np.any(distance < 0):
        raise ValueError("distance must be non-negative")
    mask000 = np.asarray(columns["overlap_mask"]).astype(str) == "000"
    if len(mask000) != len(distance):
        raise ValueError("overlap_mask and distance must have equal lengths")
    bin_specs = [(float(x), float(x + 10), f"[{x},{x + 10})") for x in range(0, 120, 10)]
    bin_specs.append((120.0, float("inf"), "[120,+inf)"))
    rows = []
    for lower, upper, label in bin_specs:
        selected = (distance >= lower) & (distance < upper)
        primary = selected & mask000
        count, denominator = int(selected.sum()), int(primary.sum())
        ci = classified["close_initial"] & primary
        cf = classified["close_final"] & primary
        flags = {"both": ci & cf, "initial_only": ci & ~cf, "final_only": ~ci & cf,
                 "neither": primary & ~ci & ~cf}
        counts = {name: int(flag.sum()) for name, flag in flags.items()}
        measures = {
            "close_initial": int(ci.sum()), "close_final": int(cf.sum()),
            "initial_only": counts["initial_only"], "final_only": counts["final_only"],
            "both": counts["both"], "either": int((ci | cf).sum()), "neither": counts["neither"],
        }
        row: dict[str, Any] = {"bin": label, "count": count, "mask000_count": denominator}
        for name, value in measures.items():
            row[f"{name}_count"] = value
            row[f"{name}_fraction_mask000"] = _safe_fraction(value, denominator)
            row[f"{name}_share_bin"] = _safe_fraction(value, count)
        row.update({f"exclusive_{name}_count": value for name, value in counts.items()})
        row["exclusive_sum_ok"] = sum(counts.values()) == denominator
        rows.append(row)
    prevalence: Counter[tuple[str, str]] = Counter()
    for component, values in (("initial", classified["relation_initial"]), ("final", classified["relation_final"])):
        prevalence.update((component, str(value)) for value in values[mask000] if value)
    category_rows = [
        {"component": component, "category": category, "count": count,
         "fraction_mask000": _safe_fraction(count, int(mask000.sum())),
         "share_all_pairs": _safe_fraction(count, len(distance))}
        for (component, category), count in sorted(prevalence.items())
    ]
    return {"profile": profile, "bins": rows, "category_prevalence": category_rows,
            "inventory": COMPONENT_INVENTORY}
