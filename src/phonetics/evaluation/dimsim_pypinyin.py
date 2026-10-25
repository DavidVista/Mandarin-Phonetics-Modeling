"""Diagnostic DIMSIM 0.2.2 scoring with strict pypinyin components.

This adapter changes only syllable decomposition (diagnostic B).  Coordinates,
weights, hardcoded-name construction, and object-identity comparisons are those
of the pinned DIMSIM implementation.  Diagnostic A remains ``dimsim.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from dimsim.utils.maps import consonantList, consonantMap_TwoDCode, vowelMap_TwoDCode
from dimsim.utils.utils import get_edit_distance_close_2d_code

from phonetics.evaluation.dimsim import DimsimDistance
from phonetics.pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, parse_components


_CANONICAL_CONSONANTS = {
    value: value for value in (*consonantList, next(key for key in consonantMap_TwoDCode if key == ""))
}


@dataclass(frozen=True)
class DimsimPinyinComponents:
    """The subset of the author's ``Pinyin`` interface used by its scorer."""

    consonant: str
    vowel: str
    tone: int

    def toStringNoTone(self) -> str:
        return "{}{}".format(self.consonant, self.vowel)

    def toStringWithTone(self) -> str:
        return "{}{}{}".format(self.consonant, self.vowel, self.tone)

    def toString(self) -> str:
        return self.toStringWithTone()


def adapted_components(token: str) -> DimsimPinyinComponents:
    """Strictly parse a TONE3 syllable and expose its DIMSIM-facing fields.

    The sole spelling conversion is ``v`` to DIMSIM's ``u:`` notation; hence
    strict ``van`` becomes ``u:an`` while ``iou``, ``uei``, and ``uen`` remain
    uncontracted.  In particular, ``hui`` is represented as ``huei`` for the
    author's hardcode lookup.  Raw strict fields remain available through
    :func:`component_mapping`.
    """

    raw = parse_components(token)
    try:
        consonant = _CANONICAL_CONSONANTS[raw.initial]
    except KeyError as exc:
        raise ValueError(f"DIMSIM has no consonant coordinate for {raw.initial!r} in {token!r}") from exc
    vowel = raw.final.replace("v", "u:")
    if vowel not in vowelMap_TwoDCode:
        raise ValueError(f"DIMSIM has no vowel coordinate for strict final {raw.final!r} in {token!r}")
    tone = int(raw.tone)
    if tone not in range(1, 6):
        raise ValueError(f"DIMSIM requires tone 1..5, found {raw.tone!r} in {token!r}")
    return DimsimPinyinComponents(consonant, vowel, tone)


def component_mapping(token: str) -> dict[str, object]:
    """Return raw strict and adapted fields for audit and test inspection."""

    raw = parse_components(token)
    adapted = adapted_components(token)
    return {
        "token": token,
        "strict_initial": raw.initial,
        "strict_final": raw.final,
        "tone": raw.tone,
        "dimsim_consonant": adapted.consonant,
        "dimsim_vowel": adapted.vowel,
        "dimsim_name": adapted.toStringNoTone(),
    }


def score_original_components(left: object, right: object) -> float:
    """Reproduce DIMSIM 0.2.2's single-syllable formula for Pinyin-like objects.

    Identity comparisons are intentional author semantics.  This function does
    not intern or repair caller-provided objects.
    """

    try:
        coordinate_distance = get_edit_distance_close_2d_code(left, right)
        factor = (
            int(left.consonant is not right.consonant)
            + int(str(left.vowel) != str(right.vowel))
            + (0.01 if left.tone is not right.tone else 0.0)
        ) / 2.1
        result = float(coordinate_distance * factor)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("objects are not supported DIMSIM Pinyin components") from exc
    if not math.isfinite(result):
        raise ValueError("DIMSIM produced a non-finite distance")
    return result


class PypinyinDimsimDistance:
    """Callable diagnostic B: strict pypinyin decomposition, original scoring."""

    def __init__(self) -> None:
        public_adapter = DimsimDistance()  # shared pinned-version/package guard
        package_hashes = public_adapter.provenance["package_file_sha256"]
        self.provenance = {
            "metric_name": "dimsim-pypinyin-components",
            "metric_version": public_adapter.provenance["metric_version"],
            "dependencies": dict(public_adapter.provenance["dependencies"]),
            "scoring_api": "dimsim.utils.utils.get_edit_distance_close_2d_code",
            "decomposition_api": "phonetics.pinyin.parse_components",
            "pinyin_contract_version": PINYIN_CONTRACT_VERSION,
            "parser_provenance": dict(PARSER_PROVENANCE),
            "canonical_weights_modified": False,
            "identity_policy": "canonical DIMSIM consonantList objects; integer tones 1..5",
            "spelling_policy": "strict final.replace('v', 'u:'); no contraction or other rewrite",
            "name_policy": "consonant + adapted vowel; hardcode keys receive that representation",
            "package_file_sha256": package_hashes,
            "adapter_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def __call__(self, left: str, right: str) -> float:
        result = score_original_components(adapted_components(left), adapted_components(right))
        if not math.isfinite(result):  # defensive boundary for callers
            raise ValueError(f"non-finite DIMSIM distance for {left!r}, {right!r}")
        return result
