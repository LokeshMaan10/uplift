"""Common interface and LightGBM factories shared by every model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


class UpliftModel(ABC):
    """Every model scores customers; higher score = target first.

    For true uplift models the score is the estimated CATE:
        P(respond | offer, x) - P(respond | no offer, x)
    Baselines (random, response) return a ranking score that is NOT an uplift,
    but they are evaluated with the same uplift metrics so we can compare.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "UpliftModel":
        ...

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        ...


def make_classifier(params: dict[str, Any], seed: int) -> LGBMClassifier:
    """LightGBM classifier for outcome models P(y=1 | x)."""
    return LGBMClassifier(**params, random_state=seed)


def make_regressor(params: dict[str, Any], seed: int) -> LGBMRegressor:
    """LightGBM regressor for effect models (pseudo-outcomes are continuous)."""
    return LGBMRegressor(**params, random_state=seed)


def proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """P(class 1) as a flat array."""
    return model.predict_proba(X)[:, 1]
