"""Pure Pinyin-inventory helpers shared by training diagnostics."""
from __future__ import annotations

from functools import lru_cache

from .pinyin import VALID_SYLLABLES, canonical_label, decompose


@lru_cache(maxsize=1)
def valid_initial_final_pairs() -> frozenset[tuple[int, int]]:
    """Return pairs represented by labels accepted by the strict Pinyin API."""
    pairs = set()
    for base in sorted(VALID_SYLLABLES):
        try:
            canonical = canonical_label(base + "1", tone_convention="explicit")
            initial_id, final_id, _ = decompose(canonical, tone_convention="explicit")
        except (ValueError, TypeError, IndexError):
            continue
        pairs.add((initial_id, final_id))
    if not pairs:
        raise RuntimeError("strict Pinyin inventory contains no valid initial/final pairs")
    return frozenset(pairs)
