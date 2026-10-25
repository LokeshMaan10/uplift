"""Module 4 — baselines that uplift models must beat.

- RandomModel:          target customers at random (what 'mail a random X%' achieves)
- ResponseModel:        classic classifier P(respond | x), ignores who got the offer
- HistoricalBuyerModel: P(respond | x, no offer) — the "propensity to buy" model
                        trained on customers who were never contacted

Both response-style models answer "who will respond?", not "who responds BECAUSE
of the offer?". Evaluating them with uplift metrics shows the gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from upliftlab.models.base import UpliftModel, make_classifier, proba


class RandomModel(UpliftModel):
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "RandomModel":
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(X))


class ResponseModel(UpliftModel):
    name = "Response model"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.model = make_classifier(params, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "ResponseModel":
        self.model.fit(X, y)  # treatment deliberately ignored
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return proba(self.model, X)


class HistoricalBuyerModel(UpliftModel):
    name = "Historical buyer"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.model = make_classifier(params, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "HistoricalBuyerModel":
        control = np.asarray(t) == 0
        self.model.fit(X[control], y[control])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return proba(self.model, X)
