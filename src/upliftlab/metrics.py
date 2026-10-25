"""Module 7 — uplift evaluation metrics.

We can't compare a prediction to a true individual uplift (it is never observed).
Instead: sort customers by predicted uplift, walk down the list, and measure the
*observed* treated-vs-control difference among the customers targeted so far.

Uplift curve  U(k) = (rate_treated(top k) - rate_control(top k)) * k
              "extra responders if we target the top k customers"
Qini curve    Q(k) = R_t(k) - R_c(k) * N_t(k) / N_c(k)
              R = responders, N = group sizes in the top k. Same idea, with the
              control count rescaled to the treated group size.
AUUC / Qini coefficient
              Area between the model's curve and the random-targeting diagonal,
              normalised by the area of a perfect model (sklift). 0 = random,
              1 = perfect, negative = worse than random.
uplift@k      treated rate - control rate among the top k% scored customers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklift.metrics import (
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_by_percentile,
    uplift_curve,
)

ArrayLike = np.ndarray | pd.Series


def uplift_metrics(y: ArrayLike, score: ArrayLike, t: ArrayLike, top_k_percents: list[int]) -> dict[str, float]:
    """Headline metrics for one model on one dataset."""
    y, score, t = np.asarray(y), np.asarray(score), np.asarray(t)
    out = {
        "qini_coef": float(qini_auc_score(y, score, t)),
        "auuc": float(uplift_auc_score(y, score, t)),
    }
    for k in top_k_percents:
        out[f"uplift@{k}%"] = float(uplift_at_k(y, score, t, strategy="overall", k=k / 100))
    return out


def bootstrap_ci(
    y: ArrayLike,
    score: ArrayLike,
    t: ArrayLike,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = qini_auc_score,
    n_iter: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, CI low, CI high) via percentile bootstrap over customers.

    Resampling the test set tells us how much the metric would move on a
    different sample of customers — the honest error bar for model comparison.
    """
    y, score, t = np.asarray(y), np.asarray(score), np.asarray(t)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        stats[i] = metric(y[idx], score[idx], t[idx])
    return float(metric(y, score, t)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def curve_points(
    y: ArrayLike, score: ArrayLike, t: ArrayLike, kind: str = "qini", n_points: int = 101
) -> pd.DataFrame:
    """Curve resampled to `n_points` on the x-axis 'fraction of customers targeted'.

    y-axis = incremental responders, expressed per customer in the dataset so
    curves from different-size datasets are comparable.
    """
    fn = {"qini": qini_curve, "uplift": uplift_curve}[kind]
    x, val = fn(np.asarray(y), np.asarray(score), np.asarray(t))
    n = len(y)
    grid = np.linspace(0, 1, n_points)
    return pd.DataFrame({"frac_targeted": grid, "incremental_per_customer": np.interp(grid, x / n, val / n)})


def decile_table(y: ArrayLike, score: ArrayLike, t: ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Observed uplift in each decile of predicted score (decile 1 = highest scores).

    A good model shows uplift falling steadily from decile 1 to 10.
    """
    table = uplift_by_percentile(np.asarray(y), np.asarray(score), np.asarray(t),
                                 strategy="overall", bins=bins, std=True)
    table = table.reset_index().rename(columns={"index": "percentile"})
    table.insert(0, "decile", range(1, len(table) + 1))
    return table
