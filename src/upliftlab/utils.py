"""Small helpers shared across modules: seeding and logging."""

from __future__ import annotations

import logging
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and hashing so results are reproducible run to run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "upliftlab", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a simple timestamped format (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
