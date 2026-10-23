"""Explicit bridge from source artifacts to the generic preparation SDK."""

from pathlib import Path
from ..artifacts import sha256_file


def create_preparation_config(*, dataset_id, source_root, wubi_path, wubi_sha256, model,
                              sampling, output_root, cache_root):
    root=Path(source_root).resolve(); corpus=root/"corpus/labelled.csv"; manifest=root/"audio/sounds_manifest.csv"
    if not corpus.is_file() or not manifest.is_file(): raise FileNotFoundError("source root is incomplete")
    if sha256_file(wubi_path) != wubi_sha256: raise ValueError("Wubi checksum mismatch")
    return {"dataset_id":dataset_id,"output_root":str(Path(output_root).resolve()),"cache_root":str(Path(cache_root).resolve()),
      "corpus":{"path":str(corpus),"sha256":sha256_file(corpus),"source_id":dataset_id,"tone_convention":"explicit"},
      "sounds":{"path":str((root/"audio/sounds").resolve()),"manifest":str(manifest),"sha256":sha256_file(manifest),"tone_convention":"explicit"},
      "wubi":{"path":str(Path(wubi_path).resolve()),"sha256":wubi_sha256},"model":dict(model),"sampling":dict(sampling)}
