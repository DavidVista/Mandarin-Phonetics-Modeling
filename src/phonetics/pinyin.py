"""Project TONE3 input contract backed by pinned pypinyin inventories."""
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import re

from pypinyin.constants import PINYIN_DICT
from pypinyin.contrib.tone_convert import to_finals, to_initials, to_normal

PINYIN_CONTRACT_VERSION = "pypinyin-tone3-1"


def _require_pypinyin_version():
    installed = version("pypinyin")
    if installed != "0.55.0":
        raise RuntimeError(f"pypinyin 0.55.0 required, found {installed}")
    return installed


_PYPINYIN_VERSION = _require_pypinyin_version()
# Dictionary readings contain tone marks; external project inputs do not.
VALID_SYLLABLES = frozenset(
    to_normal(reading, v_to_u=False).lower()
    for readings in PINYIN_DICT.values()
    for reading in readings.split(",")
)
_INPUT_SYLLABLES = tuple(sorted(base for base in VALID_SYLLABLES if re.fullmatch(r"[a-z]+", base)))
_STRICT_COMPONENTS = tuple(
    (to_initials(base, strict=True), to_finals(base, strict=True, v_to_u=False))
    for base in _INPUT_SYLLABLES
)
INITIALS = sorted({initial for initial, final in _STRICT_COMPONENTS if final})
FINALS = sorted({final for _, final in _STRICT_COMPONENTS if final})
TONES = [str(tone) for tone in range(1, 6)]
PARSER_PROVENANCE = {
    "library": "pypinyin", "version": _PYPINYIN_VERSION,
    "api": "pypinyin.contrib.tone_convert", "strict": True,
    "input_format": "lowercase-ascii-tone3", "umlaut_storage": "v",
    "input_alias": "u:", "missing_tone": "5", "empty_final": "reject",
    "inventory_source": "pypinyin.constants.PINYIN_DICT",
    "inventory_sha256": sha256("\n".join(sorted(VALID_SYLLABLES)).encode("utf-8")).hexdigest(),
    "vocabulary_sha256": sha256(repr((INITIALS, FINALS, TONES)).encode("utf-8")).hexdigest(),
}


@dataclass(frozen=True)
class PinyinComponents:
    initial: str
    final: str
    tone: str


def canonical_label(token: str, *, tone_convention="neutral_unmarked") -> str:
    """Accept one lowercase TONE3 syllable, allowing u: and unmarked neutral."""
    if tone_convention not in {"neutral_unmarked", "explicit"}:
        raise ValueError(f"unsupported tone convention: {tone_convention!r}")
    if not isinstance(token, str) or not token:
        raise ValueError(f"Pinyin must be a non-empty string: {token!r}")
    if re.fullmatch(r"(?:[a-z]|u:)+[1-5]?", token) is None:
        raise ValueError(f"invalid Pinyin format (lowercase ASCII TONE3 required): {token!r}")
    canonical = token.replace("u:", "v")
    if canonical[-1] not in TONES:
        if tone_convention == "explicit":
            raise ValueError(f"explicit tone required: {token!r}")
        canonical += "5"
    if re.fullmatch(r"[a-z]+[1-5]", canonical) is None:
        raise ValueError(f"invalid canonical Pinyin format: {token!r}")
    base = canonical[:-1]
    if base not in VALID_SYLLABLES:
        raise ValueError(f"syllable absent from pypinyin dictionary: {token!r}")
    initial = to_initials(base, strict=True)
    final = to_finals(base, strict=True, v_to_u=False)
    if not final or initial not in INITIALS or final not in FINALS:
        raise ValueError(f"unsupported strict Pinyin components (non-empty final required): {token!r}")
    return canonical


def validate_canonical_label(token: str) -> str:
    """Reject noncanonical stored labels instead of silently repairing them."""
    if canonical_label(token, tone_convention="explicit") != token:
        raise ValueError(f"stored Pinyin label is not canonical: {token!r}")
    return token


def parse_components(token: str, *, tone_convention="neutral_unmarked") -> PinyinComponents:
    canonical = canonical_label(token, tone_convention=tone_convention)
    base = canonical[:-1]
    return PinyinComponents(to_initials(base, strict=True), to_finals(base, strict=True, v_to_u=False), canonical[-1])


def encode_components(components: PinyinComponents) -> tuple[int, int, int]:
    try:
        return INITIALS.index(components.initial), FINALS.index(components.final), TONES.index(components.tone)
    except ValueError as exc:
        raise ValueError(f"components outside Pinyin vocabulary: {components!r}") from exc


def canonical_labels(tokens, *, tone_convention="neutral_unmarked"):
    return [canonical_label(token, tone_convention=tone_convention) for token in tokens]


def decompose(token: str, *, tone_convention="neutral_unmarked") -> tuple[int, int, int]:
    return encode_components(parse_components(token, tone_convention=tone_convention))
