"""Reproducibility helpers with independent random streams."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, stream: int = 0) -> torch.Generator:
    """Return an independent generator; module initialization cannot advance it."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1_000_003 * int(stream))
    return generator
