"""Model-independent and model-specific tools for representation evaluation."""

__all__ = ["DimsimDistance", "load_evaluation_bundle", "prepare_evaluation_bundle"]

def __getattr__(name):
    if name in {"load_evaluation_bundle", "prepare_evaluation_bundle"}:
        from .bundle import load_evaluation_bundle, prepare_evaluation_bundle
        return {"load_evaluation_bundle":load_evaluation_bundle,"prepare_evaluation_bundle":prepare_evaluation_bundle}[name]
    if name == "DimsimDistance":
        from .dimsim import DimsimDistance
        return DimsimDistance
    raise AttributeError(name)
