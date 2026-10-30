"""Versioned hierarchical anchor-Q triplet sampling."""

from __future__ import annotations

import hashlib
import json

import numpy as np

SAMPLER_VERSION = 1
TRIPLET_SCHEMA_VERSION = 1
SEED = 20261026
EDGES = (
    ("initial", "000", "100"), ("initial", "001", "101"),
    ("initial", "010", "110"), ("initial", "011", "111"),
    ("final", "000", "010"), ("final", "001", "011"),
    ("final", "100", "110"), ("final", "101", "111"),
    ("tone", "000", "001"), ("tone", "010", "011"),
    ("tone", "100", "101"), ("tone", "110", "111"),
)


def _stable_id(prefix: str, values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()


def sample_anchor_q(occurrences: list[dict], syllables: list[dict], *, draws_per_edge: int = 1000,
                    seed: int = SEED) -> tuple[list[dict], dict]:
    """Generate fixed draws; diagnostics are returned even for coverage failures."""
    if isinstance(draws_per_edge, bool) or not isinstance(draws_per_edge, int) or draws_per_edge <= 0:
        raise ValueError("draws_per_edge must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("anchor-Q seed must be a non-negative integer")
    by_syllable = {row["syllable_id"]: row for row in syllables}
    occurrence_map = {row["occurrence_id"]: row for row in occurrences}
    members = {key: sorted(row["occurrence_ids"]) for key, row in by_syllable.items()}
    def overlap(anchor_id: str, candidate_id: str) -> str:
        return "".join(str(int(by_syllable[anchor_id][key] == by_syllable[candidate_id][key]))
                       for key in ("initial", "final", "tone"))
    diagnostics = {"sampler_version": SAMPLER_VERSION, "seed": seed, "rng": "numpy.random.PCG64",
                   "draws_per_edge": draws_per_edge, "edges": {}}
    supports = {}
    candidate_supports = {}
    for component, low_mask, high_mask in EDGES:
        edge_id = f"{component}:{low_mask}->{high_mask}"
        eligible, excluded = [], {}
        for sid in sorted(by_syllable):
            ids = members[sid]
            if not ids:
                excluded[sid] = "no_occurrences"; continue
            if high_mask == "111" and len(ids) < 2:
                excluded[sid] = "high_111_requires_two_occurrences"; continue
            high = [x for x in sorted(by_syllable) if overlap(sid, x) == high_mask and (x != sid or high_mask == "111")]
            low = [x for x in sorted(by_syllable) if overlap(sid, x) == low_mask and x != sid]
            if not high or not low:
                excluded[sid] = "missing_high_or_low_syllable"; continue
            eligible.append(sid)
            candidate_supports[(component, low_mask, high_mask, sid)] = (high, low)
        supports[edge_id] = (eligible, excluded)
        diagnostics["edges"][edge_id] = {"eligible_anchor_syllables": len(eligible),
                                          "excluded_anchor_syllables": len(excluded),
                                          "exclusion_reasons": excluded}
    missing = [edge for edge, (eligible, _) in supports.items() if not eligible]
    if missing:
        diagnostics["missing_edges"] = missing
        raise ValueError("anchor-Q coverage failure; diagnostics=" + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
    rng = np.random.Generator(np.random.PCG64(seed))
    output = []
    for component, low_mask, high_mask in EDGES:
        edge_id = f"{component}:{low_mask}->{high_mask}"
        anchors = supports[edge_id][0]
        for draw_index in range(draws_per_edge):
            anchor_sid = anchors[int(rng.integers(len(anchors)))]
            anchor_occ = members[anchor_sid][int(rng.integers(len(members[anchor_sid])))]
            eligible_high, eligible_low = candidate_supports[(component, low_mask, high_mask, anchor_sid)]
            high_sid = eligible_high[int(rng.integers(len(eligible_high)))]
            low_sid = eligible_low[int(rng.integers(len(eligible_low)))]
            high_occ_choices = [x for x in members[high_sid] if x != anchor_occ]
            high_occ = high_occ_choices[int(rng.integers(len(high_occ_choices)))]
            low_occ_choices = members[low_sid]
            low_occ = low_occ_choices[int(rng.integers(len(low_occ_choices)))]
            if len({anchor_occ, high_occ, low_occ}) != 3:
                raise ValueError(f"anchor-Q sampler produced repeated occurrence IDs for {edge_id}")
            triple = [anchor_occ, high_occ, low_occ]
            triplet_id = _stable_id("triplet:", [edge_id, *triple])
            output.append({"schema_version": TRIPLET_SCHEMA_VERSION, "sampler_version": SAMPLER_VERSION,
                           "edge_id": edge_id, "component": component, "low_mask": low_mask, "high_mask": high_mask,
                           "draw_index": draw_index, "draw_id": f"anchor-q:v1:{edge_id}:{draw_index:06d}",
                           "triplet_id": triplet_id, "anchor_syllable_id": anchor_sid,
                           "high_syllable_id": high_sid, "low_syllable_id": low_sid,
                           "anchor_id": anchor_occ, "high_id": high_occ, "low_id": low_occ})
    diagnostics["draw_count"] = len(output)
    diagnostics["unique_triplets_by_edge"] = {edge: len({r["triplet_id"] for r in output if r["edge_id"] == edge}) for edge in supports}
    diagnostics["unique_anchor_occurrences_by_edge"] = {edge: len({r["anchor_id"] for r in output if r["edge_id"] == edge}) for edge in supports}
    diagnostics["unique_anchor_syllables_by_edge"] = {edge: len({r["anchor_syllable_id"] for r in output if r["edge_id"] == edge}) for edge in supports}
    diagnostics["unique_occurrences_by_edge"] = {edge: len({r[k] for r in output if r["edge_id"] == edge for k in ("anchor_id", "high_id", "low_id")}) for edge in supports}
    for edge in supports:
        diagnostics["edges"][edge].update({
            "draw_count": sum(row["edge_id"] == edge for row in output),
            "unique_triplets": diagnostics["unique_triplets_by_edge"][edge],
            "unique_anchor_occurrences": diagnostics["unique_anchor_occurrences_by_edge"][edge],
            "unique_anchor_syllables": diagnostics["unique_anchor_syllables_by_edge"][edge],
            "unique_occurrences": diagnostics["unique_occurrences_by_edge"][edge],
        })
    return output, diagnostics
