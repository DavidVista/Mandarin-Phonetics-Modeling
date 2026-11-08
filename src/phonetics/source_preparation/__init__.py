"""Pinned DuYu export and safe local sound archive helpers."""

from importlib import import_module

_EXPORTS = {"pack_sounds": ".archive", "unpack_sounds": ".archive", "export_duyu": ".duyu",
            "CHINESE_LESSONS_ALIASES": ".sound_manifest", "build_sound_manifest": ".sound_manifest",
            "download_sounds": ".download", "validate_source_bundle": ".download"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name], __name__), name)
