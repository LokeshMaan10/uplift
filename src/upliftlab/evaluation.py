"""Cross-validation for uplift models (stratified on treatment x outcome)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from upliftlab.metrics import uplift_metrics
from upliftlab.models import UpliftModel


def cross_validate(
    factory: Callable[[], UpliftModel],
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    folds: int,
    seed: int,
    top_k_percents: list[int],
) -> pd.DataFrame:
    """Fit a fresh model on k-1 folds, score the held-out fold; one row per fold.

    Stratifying on (treatment, outcome) keeps every fold's treated share and
    response rate equal, so fold metrics differ only because of the model.
    """
    strata = np.asarray(t) * 2 + np.asarray(y)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, strata)):
        model = factory().fit(X.iloc[tr], y.iloc[tr], t.iloc[tr])
        score = model.predict_uplift(X.iloc[va])
        rows.append({"fold": fold, **uplift_metrics(y.iloc[va], score, t.iloc[va], top_k_percents)})
    return pd.DataFrame(rows)


def summarize_cv(cv_scores: pd.DataFrame, metric: str = "qini_coef") -> dict[str, float]:
    """Mean and std of a metric across folds."""
    return {f"cv_{metric}_mean": float(cv_scores[metric].mean()), f"cv_{metric}_std": float(cv_scores[metric].std())}
