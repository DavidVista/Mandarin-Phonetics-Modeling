"""Deterministic sampling of unique unordered pairs without quadratic materialization."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


def pair_count(item_count: int) -> int:
    """Return the number of unique unordered pairs of ``item_count`` objects."""
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    return item_count * (item_count - 1) // 2


def pair_from_rank(item_count: int, rank: int) -> tuple[int, int]:
    """Map a rank in lexicographic upper-triangle order to ``(left, right)``."""
    total = pair_count(item_count)
    if rank < 0 or rank >= total:
        raise IndexError(f"pair rank {rank} outside [0, {total})")

    # Number of pairs before row i: i * (2*n-i-1) // 2. Binary search avoids
    # floating-point inversion errors for large collections.
    low, high = 0, item_count - 1
    while low + 1 < high:
        middle = (low + high) // 2
        before = middle * (2 * item_count - middle - 1) // 2
        if before <= rank:
            low = middle
        else:
            high = middle
    left = low
    before = left * (2 * item_count - left - 1) // 2
    right = left + 1 + (rank - before)
    return left, right


def iter_all_pairs(item_count: int, *, max_pairs: int) -> Iterator[tuple[int, int]]:
    """Yield all unique pairs, rejecting an unexpectedly large expansion."""
    total = pair_count(item_count)
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if total > max_pairs:
        raise ValueError(f"all-pairs plan has {total} pairs, above cap {max_pairs}")
    for left in range(item_count):
        for right in range(left + 1, item_count):
            yield left, right


def sample_pair_indices(item_count: int, sample_count: int, seed: int) -> list[tuple[int, int]]:
    """Uniformly sample unique unordered pairs without constructing all pairs."""
    total = pair_count(item_count)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count > total:
        raise ValueError(f"requested {sample_count} pairs from population of {total}")
    ranks = random.Random(seed).sample(range(total), sample_count)
    return [pair_from_rank(item_count, rank) for rank in sorted(ranks)]


def make_pair_plan(
    item_ids: Sequence[str], *, sample_count: int | None, seed: int, max_all_pairs: int
) -> list[tuple[str, str]]:
    """Return canonical ID pairs from an already stably ordered collection."""
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item_ids must be unique")
    if list(item_ids) != sorted(item_ids):
        raise ValueError("item_ids must be stably sorted")
    if sample_count is None:
        indices = iter_all_pairs(len(item_ids), max_pairs=max_all_pairs)
    else:
        indices = sample_pair_indices(len(item_ids), sample_count, seed)
    return [(item_ids[left], item_ids[right]) for left, right in indices]
