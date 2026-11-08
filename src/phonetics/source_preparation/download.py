"""Pinned, atomic downloads of the official Mandarin sound archive."""
from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

from ..artifacts import sha256_file, write_json
from .archive import _validate_entries

CHINESE_LESSONS_URL = "https://www.chinese-lessons.com/sounds/Mandarin_sounds.zip"
CHINESE_LESSONS_SHA256 = "be17ce405fcfd6c63a940fa9ac2a7985aa3e6e464543d72f36f2b9dc6ee8b8cd"
# The official server rejects urllib's default User-Agent with HTTP 406.
DOWNLOAD_HEADERS = {"User-Agent": "curl/8.0.0", "Accept": "*/*"}
ARCHIVE_NAME = "Mandarin_sounds.zip"


def validate_source_bundle(path, *, url=CHINESE_LESSONS_URL, expected_sha256=CHINESE_LESSONS_SHA256):
    """Verify complete source content and the requested source identity."""
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source bundle must be a non-symlink directory")
    if {p.name for p in root.iterdir()} != {ARCHIVE_NAME, "manifest.json"}:
        raise ValueError("source bundle artifact set mismatch")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("source manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _source_manifest(url, expected_sha256)
    if manifest != expected:
        raise ValueError("source bundle configuration mismatch; use a new output ID")
    validate_sound_archive(root / ARCHIVE_NAME, expected_sha256)
    return root


def _source_manifest(url, digest):
    return {"schema_version": 1, "bundle_type": "sound_source", "source_url": url,
            "source_page": "https://www.chinese-lessons.com/download.htm",
            "attribution": "Chinese-Lessons.com, Mandarin Sounds",
            "license_url": "https://creativecommons.org/licenses/by-nc-nd/3.0/us/",
            "request_headers": dict(DOWNLOAD_HEADERS), "artifacts": {ARCHIVE_NAME: digest}}


def validate_sound_archive(path, expected_sha256):
    """Validate bytes, safe flat filenames and every entry's CRC."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("archive must be a regular, non-symlink file")
    if sha256_file(path) != expected_sha256:
        raise ValueError("archive checksum mismatch; use a new output for changed inputs")
    with zipfile.ZipFile(path) as archive:
        _validate_entries(archive.infolist())
        if archive.testzip() is not None:
            raise ValueError("archive contains a corrupt entry")
    return path


def download_sounds(destination, *, url=CHINESE_LESSONS_URL,
                    expected_sha256=CHINESE_LESSONS_SHA256, timeout=60):
    """Download or reuse a verified directory containing ZIP and provenance.

    This validates the archive container; decoded-audio checks belong to the
    sound inventory stage. No embeddings or ML dependencies are imported.
    """
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("sound download requires an HTTPS URL")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256")
    if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be positive and at most 3600 seconds")
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        return validate_source_bundle(target, url=url, expected_sha256=expected_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent))
    try:
        with (temporary / ARCHIVE_NAME).open("wb") as output:
            request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200 or not response.geturl().startswith("https://"):
                    raise ValueError("download did not return a successful HTTPS response")
                count = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    count += len(block)
                declared = response.headers.get("Content-Length")
                if declared is not None and count != int(declared):
                    raise ValueError("incomplete sound archive response")
        write_json(temporary / "manifest.json", _source_manifest(url, expected_sha256))
        validate_source_bundle(temporary, url=url, expected_sha256=expected_sha256)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(temporary, target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
