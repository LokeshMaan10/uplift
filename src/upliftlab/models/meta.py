"""Module 5 — meta-learners: turn any ML model (here LightGBM) into a CATE estimator.

Notation: y outcome, t treatment (0/1), x features, e = P(t=1) propensity.
Hillstrom/Criteo are randomized, so e is a known constant (the treated share).

S-learner  One model f(x, t). uplift = f(x, 1) - f(x, 0).
           Simple, but the tree may barely use t -> uplift shrunk toward 0.
T-learner  Two models: mu1 on treated, mu0 on control. uplift = mu1(x) - mu0(x).
           Each model is noisy on its own; their difference adds the noise up.
X-learner  T-learner, then impute individual effects and learn them directly:
             treated:  D1 = y - mu0(x)     control: D0 = mu1(x) - y
           tau1 fits D1, tau0 fits D0;  uplift = e * tau0(x) + (1 - e) * tau1(x).
           Good when groups are unbalanced (e.g. Criteo, 85% treated).
R-learner  Robinson decomposition: y - m(x) = tau(x) * (t - e) + noise, m = E[y|x].
           Fit tau by weighted regression of (y - m)/(t - e) with weights (t - e)^2,
           using out-of-fold m(x) ("cross-fitting") to avoid overfitting bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from upliftlab.models.base import UpliftModel, make_classifier, make_regressor, proba

T_COL = "treatment_flag"


class SLearner(UpliftModel):
    name = "S-learner"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.model = make_classifier(params, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "SLearner":
        self.model.fit(X.assign(**{T_COL: np.asarray(t)}), y)
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(P(y=1 | no offer), P(y=1 | offer)) for each row."""
        return proba(self.model, X.assign(**{T_COL: 0})), proba(self.model, X.assign(**{T_COL: 1}))

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_outcomes(X)
        return p1 - p0


class TLearner(UpliftModel):
    name = "T-learner"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.mu0 = make_classifier(params, seed)
        self.mu1 = make_classifier(params, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "TLearner":
        t = np.asarray(t)
        self.mu0.fit(X[t == 0], y[t == 0])
        self.mu1.fit(X[t == 1], y[t == 1])
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return proba(self.mu0, X), proba(self.mu1, X)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_outcomes(X)
        return p1 - p0


class XLearner(UpliftModel):
    name = "X-learner"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        self.stage1 = TLearner(params, seed)
        self.tau0 = make_regressor(params, seed)
        self.tau1 = make_regressor(params, seed)
        self.e = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "XLearner":
        t, y_arr = np.asarray(t), np.asarray(y)
        self.e = float(t.mean())
        self.stage1.fit(X, y, t)
        X1, X0 = X[t == 1], X[t == 0]
        d1 = y_arr[t == 1] - proba(self.stage1.mu0, X1)  # what treated gained vs predicted control
        d0 = proba(self.stage1.mu1, X0) - y_arr[t == 0]  # what control missed vs predicted treated
        self.tau1.fit(X1, d1)
        self.tau0.fit(X0, d0)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


class RLearner(UpliftModel):
    name = "R-learner"

    def __init__(self, params: dict[str, Any], seed: int = 0, folds: int = 5) -> None:
        self.params, self.seed, self.folds = params, seed, folds
        self.tau = make_regressor(params, seed)
        self.e = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "RLearner":
        t, y_arr = np.asarray(t), np.asarray(y)
        self.e = float(t.mean())
        cv = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        m_hat = cross_val_predict(make_classifier(self.params, self.seed), X, y_arr, cv=cv,
                                  method="predict_proba")[:, 1]  # out-of-fold E[y | x]
        t_res, y_res = t - self.e, y_arr - m_hat
        self.tau.fit(X, y_res / t_res, sample_weight=t_res**2)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.tau.predict(X)
