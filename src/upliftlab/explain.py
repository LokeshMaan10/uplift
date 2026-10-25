"""Module 9 — explain WHY the model thinks a customer is persuadable.

Uplift models such as the X-learner combine several sub-models, so there is no
single tree to explain. Standard trick: fit a "surrogate" LightGBM regressor
that mimics the predicted uplift from the features, check it mimics well (R^2),
then run SHAP on the surrogate. SHAP values then read as:
    "this feature pushed this customer's predicted uplift up/down by X points".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


def fit_surrogate(
    X: pd.DataFrame, uplift: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[LGBMRegressor, float]:
    """Regressor X -> predicted uplift, plus its in-sample R^2 (fidelity)."""
    model = LGBMRegressor(**params, random_state=seed).fit(X, uplift)
    return model, float(r2_score(uplift, model.predict(X)))


def shap_values(model: LGBMRegressor, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(SHAP matrix [n_rows, n_features], expected value) for a tree model."""
    explainer = shap.TreeExplainer(model)
    return np.asarray(explainer.shap_values(X)), float(np.ravel(explainer.expected_value)[0])


def global_importance(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature, plus direction.

    direction = correlation between the feature's value and its SHAP value:
    +1 'higher value -> more persuadable', -1 'higher value -> less persuadable'.
    """
    rows = []
    for i, col in enumerate(X.columns):
        x = X[col].to_numpy(dtype=float)
        s = values[:, i]
        corr = np.corrcoef(x, s)[0, 1] if x.std() > 0 and s.std() > 0 else 0.0
        rows.append({"feature": col, "mean_abs_shap": float(np.abs(s).mean()), "direction": float(corr)})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def local_explanation(values_row: np.ndarray, x_row: pd.Series, top_n: int = 6) -> pd.DataFrame:
    """Top features pushing ONE customer's uplift up or down."""
    df = pd.DataFrame({"feature": x_row.index, "value": x_row.values, "shap": values_row})
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
