"""Occurrence-only DIMSIM sampling plans for schema-3 bundles."""

from __future__ import annotations

import hashlib
import json
import numpy as np

from .sampling import sample_pair_indices

SAMPLER_VERSION = 1
PAIR_SCHEMA_VERSION = 1


def pair_identity(left: str, right: str) -> str:
    endpoints = sorted((left, right))
    raw = json.dumps(endpoints, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "pair:" + hashlib.sha256(raw).hexdigest()


def sample_natural(occurrences: list[dict], count: int = 12000, seed: int = 42) -> list[tuple[str, str, str, int]]:
    ids = sorted(row["occurrence_id"] for row in occurrences if row["initial"])
    population = len(ids) * (len(ids) - 1) // 2
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > population:
        raise ValueError(f"natural DIMSIM sample_count {count} exceeds eligible unique pair population {population}")
    # Rank sampling avoids materializing the quadratic population and preserves
    # the project's established seeded natural-pair ordering.
    selected = sample_pair_indices(len(ids), count, seed)
    return [(ids[i], ids[j], pair_identity(ids[i], ids[j]), index)
            for index, (i, j) in enumerate(selected)]


def sample_balanced(occurrences: list[dict], syllables: list[dict], count: int = 12000,
                    seed: int = 20261026) -> list[tuple[str, str, str, int, str, str]]:
    eligible = {row["syllable_id"]: sorted(row["occurrence_ids"]) for row in syllables if row["initial"]}
    ids = sorted(eligible)
    if not ids:
        raise ValueError("balanced DIMSIM has no nonzero-initial syllables")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for index in range(count):
        first_sid = ids[int(rng.integers(len(ids)))]
        seconds = [sid for sid in ids if sid != first_sid or len(eligible[sid]) >= 2]
        if not seconds: raise ValueError("balanced DIMSIM coverage failure: no eligible second syllable")
        second_sid = seconds[int(rng.integers(len(seconds)))]
        first_id = eligible[first_sid][int(rng.integers(len(eligible[first_sid])))]
        second_choices = [x for x in eligible[second_sid] if x != first_id]
        second_id = second_choices[int(rng.integers(len(second_choices)))]
        result.append((first_id, second_id, pair_identity(first_id, second_id), index, first_sid, second_sid))
    return result
