"""Download and verify the pinned BERT files for training or evaluation."""
import os
from pathlib import Path
from threading import RLock

from .architecture import validate_bert_snapshot
from .artifacts import safe_artifact, sha256_file
from .bert import BertInputs
from .bundles import staging_directory


_SNAPSHOT_ENV = "PHONETICS_BERT_SNAPSHOT"
_SNAPSHOT_LOCK = RLock()


def _download_snapshot(spec, *, pretrained):
    from huggingface_hub import snapshot_download

    files = {name: digest for name, digest in spec["files"].items()
             if pretrained or name != "model.safetensors"}
    cache = Path(os.environ.get("PHONETICS_BERT_CACHE") or Path.home() / ".cache" / "phonetics" / "bert")
    revision = cache.expanduser().resolve() / spec["revision"]
    full = revision / "pretrained"
    target = full if pretrained or full.is_dir() else revision / "tokenizer"
    if not target.exists():
        try:
            with staging_directory(target) as stage:
                snapshot_download(repo_id=spec["model_id"], revision=spec["revision"],
                                  allow_patterns=list(files), local_dir=stage)
                for name, digest in files.items():
                    if sha256_file(safe_artifact(stage, name)) != digest:
                        raise ValueError(f"BERT snapshot checksum mismatch: {name}")
        except FileExistsError:
            if not target.is_dir():
                raise
    return target


def load_bert_inputs(spec, *, pretrained=False):
    """Use an explicit snapshot or cache the required files from the pinned revision."""
    validate_bert_snapshot(spec)
    with _SNAPSHOT_LOCK:
        if os.environ.get(_SNAPSHOT_ENV):
            return BertInputs(spec, pretrained=pretrained)
        previous = os.environ.get(_SNAPSHOT_ENV)
        os.environ[_SNAPSHOT_ENV] = str(_download_snapshot(spec, pretrained=pretrained))
        try:
            return BertInputs(spec, pretrained=pretrained)
        finally:
            if previous is None:
                os.environ.pop(_SNAPSHOT_ENV, None)
            else:
                os.environ[_SNAPSHOT_ENV] = previous
