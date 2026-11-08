"""Model-independent audio inventory built before acoustic inference."""
from __future__ import annotations

from pathlib import Path
import wave

from .artifacts import read_jsonl, sha256_file, write_jsonl
from .bundles import seal_bundle, staging_directory, validate_bundle
from .pinyin import PARSER_PROVENANCE, canonical_label, validate_canonical_label
from .source_preparation.archive import unpack_sounds
from .source_preparation.download import ARCHIVE_NAME, validate_source_bundle
from .source_preparation.aliases import CHINESE_LESSONS_ALIASES


def inspect_audio(path):
    """Read every decoded frame, rather than trusting only container headers."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            rate, channels, frames = handle.getframerate(), handle.getnchannels(), handle.getnframes()
            data = handle.readframes(frames)
            if len(data) != frames * channels * handle.getsampwidth():
                raise ValueError(f"truncated audio: {path.name}")
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Install the audio dependency group to decode MP3 inputs") from exc
        with sf.SoundFile(path) as handle:
            rate, channels, frames = handle.samplerate, handle.channels, len(handle)
            read_frames = sum(len(block) for block in handle.blocks(blocksize=65536))
            if read_frames != frames:
                raise ValueError(f"truncated audio: {path.name}")
    if rate <= 0 or channels <= 0 or frames <= 0:
        raise ValueError(f"empty or invalid audio: {path.name}")
    return {"sample_rate": rate, "channels": channels, "frames": frames}


def prepare_sound_inventory(source_bundle, destination):
    source = validate_source_bundle(source_bundle)
    identity = {"source_manifest_sha256": sha256_file(source / "manifest.json"),
                "parser": PARSER_PROVENANCE, "aliases": CHINESE_LESSONS_ALIASES,
                "tone_convention": "neutral_unmarked", "contract": "sound-inventory-1"}
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        validate_sound_inventory(target, identity=identity)
        return target
    with staging_directory(target) as temporary:
        unpack_sounds(source / ARCHIVE_NAME, temporary / "sounds",
                      expected_sha256=sha256_file(source / ARCHIVE_NAME))
        records, rejected = [], []
        seen = set()
        for path in sorted((temporary / "sounds").iterdir()):
            # Validate decoding even for filenames outside the Pinyin domain.
            info = inspect_audio(path)
            raw = CHINESE_LESSONS_ALIASES.get(path.name, path.stem)
            try:
                label = canonical_label(raw, tone_convention="neutral_unmarked")
            except ValueError as exc:
                rejected.append({"file": path.name, "reason": "unsupported_pinyin", "detail": str(exc)})
                continue
            if label in seen:
                raise ValueError(f"duplicate canonical sound label: {label}")
            seen.add(label)
            records.append({"label": label, "relative_path": f"sounds/{path.name}",
                            "sha256": sha256_file(path), **info})
        if not records:
            raise ValueError("sound inventory is empty")
        write_jsonl(temporary / "inventory.jsonl", sorted(records, key=lambda r: r["label"]))
        write_jsonl(temporary / "rejections.jsonl", rejected)
        seal_bundle(temporary, "sound_inventory", identity)
        validate_sound_inventory(temporary)
    return target


def validate_sound_inventory(path, *, identity=None):
    manifest = validate_bundle(path, "sound_inventory", identity=identity)
    rows = read_jsonl(Path(path) / "inventory.jsonl")
    labels = [validate_canonical_label(row["label"]) for row in rows]
    if not labels or labels != sorted(set(labels)):
        raise ValueError("sound labels must be nonempty, sorted and unique")
    from .artifacts import safe_artifact
    for row in rows:
        audio = safe_artifact(path, row["relative_path"])
        if sha256_file(audio) != row["sha256"]:
            raise ValueError("inventory audio checksum mismatch")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("sample_rate", "channels", "frames")):
            raise ValueError("invalid inventory audio dimensions")
    return manifest
