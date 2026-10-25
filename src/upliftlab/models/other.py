"""Module 6 — other uplift approaches.

Class transformation (Jaskowski & Jaroszewicz, 2012)
    Define Z = 1 if (treated AND responded) or (control AND did not respond).
    With a 50/50 split:  uplift(x) = 2 * P(Z=1 | x) - 1.
    One ordinary classifier does the job. For unbalanced splits we reweight
    treated/control to 50/50 so the identity still holds.

Uplift random forest (causalml)
    Trees whose splits maximise the DIFFERENCE between treated and control
    response distributions (here KL divergence), instead of purity of y.

Causal forest (econml CausalForestDML)
    "Double ML" + honest forest: first removes what x predicts about y and t
    (nuisance models, cross-fitted), then grows a forest on the residuals to
    estimate tau(x). Gives statistically grounded CATE estimates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from upliftlab.models.base import UpliftModel, make_classifier, make_regressor, proba


class ClassTransformation(UpliftModel):
    name = "Class transformation"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.model = make_classifier(params, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "ClassTransformation":
        t, y_arr = np.asarray(t), np.asarray(y)
        e = t.mean()
        z = (y_arr == t).astype(int)
        weights = np.where(t == 1, 0.5 / e, 0.5 / (1 - e))  # rebalance to a 50/50 split
        self.model.fit(X, z, sample_weight=weights)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return 2 * proba(self.model, X) - 1


class UpliftRandomForest(UpliftModel):
    name = "Uplift RF (KL)"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.params, self.seed = params, seed
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "UpliftRandomForest":
        from causalml.inference.tree import UpliftRandomForestClassifier

        self.model = UpliftRandomForestClassifier(
            control_name="0", max_features=X.shape[1], random_state=self.seed, n_jobs=-1, **self.params
        )
        self.model.fit(X.to_numpy(dtype=float), np.asarray(t).astype(str), np.asarray(y))
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy(dtype=float))[:, 0]


class CausalForest(UpliftModel):
    name = "Causal forest"

    def __init__(self, params: dict[str, Any], lgbm_params: dict[str, Any], seed: int = 0) -> None:
        self.params, self.lgbm_params, self.seed = params, lgbm_params, seed
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "CausalForest":
        from econml.dml import CausalForestDML

        p = dict(self.params)
        self.model = CausalForestDML(
            model_y=make_regressor(self.lgbm_params, self.seed),
            model_t=make_classifier(self.lgbm_params, self.seed),
            discrete_treatment=True,
            cv=p.pop("cv", 3),
            random_state=self.seed,
            **p,
        )
        self.model.fit(np.asarray(y, dtype=float), np.asarray(t), X=X.to_numpy(dtype=float))
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.effect(X.to_numpy(dtype=float)).ravel()
