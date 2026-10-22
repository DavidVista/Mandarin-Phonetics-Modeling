"""Versioned historical pinyin rules; no implicit spelling corrections."""

PINYIN_CONTRACT_VERSION = "legacy-2"

INITIALS = ["", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"]

FINALS = ["a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v", "ve", "van", "vn"]

TONES = ["1", "2", "3", "4", "5"]

ZERO_INITIAL = {"yi":"i","ya":"ia","yao":"iao","ye":"ie","you":"iu","yan":"ian","yin":"in","yang":"iang","ying":"ing","yong":"iong","yu":"v","yue":"ve","yuan":"van","yun":"vn","wu":"u","wa":"ua","wo":"uo","wai":"uai","wei":"ui","wan":"uan","wen":"un","wang":"uang","weng":"ong"}

def _canonical_labels(tokens: list[str]) -> list[str]:
    return [t if t[-1:].isdigit() else t + "5" for t in tokens]

def _decompose(token: str) -> tuple[int, int, int]:
    tone = token[-1] if token[-1:].isdigit() else "5"
    base = token[:-1] if token[-1:].isdigit() else token
    base = ZERO_INITIAL.get(base, base)
    initial = next((x for x in sorted(INITIALS[1:], key=len, reverse=True) if base.startswith(x)), "")
    final = base[len(initial):]
    if initial in {"j", "q", "x"}:
        final = {"u":"v", "ue":"ve", "uan":"van", "un":"vn"}.get(final, final)
    if final in FINALS and tone in TONES:
        return INITIALS.index(initial), FINALS.index(final), TONES.index(tone)
    raise ValueError(f"invalid pinyin decomposition: {token}")

def canonical_labels(tokens, *, tone_convention="neutral_unmarked"):
    if tone_convention not in {"neutral_unmarked", "explicit"}:
        raise ValueError("unsupported tone convention")
    if any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("pinyin tokens must be non-empty strings")
    if tone_convention == "explicit" and any(t[-1:] not in TONES for t in tokens):
        raise ValueError("explicit tone required")
    return _canonical_labels(tokens)

decompose = _decompose
