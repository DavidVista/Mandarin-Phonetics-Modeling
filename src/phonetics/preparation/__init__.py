"""Public prepared-dataset API."""

from importlib import import_module

_EXPORTS = {"prepare_dataset": ".pipeline", "validate_prepared": ".pipeline",
            "prepare_acoustic_bundle": ".acoustic_bundle", "validate_acoustic_bundle": ".acoustic_bundle",
            "assemble_dataset": ".assembly", "prepare_corpus": ".corpus_bundle",
            "validate_corpus_bundle": ".corpus_bundle"}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name], __name__), name)

__all__ = ["assemble_dataset", "prepare_acoustic_bundle", "validate_acoustic_bundle", "prepare_corpus",
           "validate_corpus_bundle", "prepare_dataset", "validate_prepared"]
