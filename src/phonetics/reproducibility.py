"""Deterministic training seeds and serializable RNG snapshots."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def make_generator(seed: int, stream: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1_000_003 * int(stream))
    return generator


def rng_snapshot(loader_generator: torch.Generator | None = None) -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": list(random.getstate()),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if loader_generator is not None:
        state["dataloader"] = loader_generator.get_state()
    return state
