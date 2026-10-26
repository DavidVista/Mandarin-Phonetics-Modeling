"""Research orchestration APIs with no compute at import time."""
__all__=["GradualityConfig","run_graduality"]
def __getattr__(name):
    if name in __all__:
        from .wav2vec2_graduality import GradualityConfig,run_graduality
        return {"GradualityConfig":GradualityConfig,"run_graduality":run_graduality}[name]
    raise AttributeError(name)
