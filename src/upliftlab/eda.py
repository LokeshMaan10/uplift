"""Statistics for Module 2: randomization checks and the average treatment effect.

Why these checks matter: uplift models learn "treated minus control" differences.
That is only a *causal* effect if treatment was assigned at random, i.e. the
two groups look the same before the offer. We test that three ways:
    1. Standardized mean difference (SMD) per feature   -> |SMD| < 0.1 is balanced
    2. Welch t-test per feature                         -> no systematic small p-values
    3. Propensity AUC: can a model predict treatment?   -> ~0.5 means "no"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SMD_THRESHOLD = 0.1  # common rule of thumb (Austin, 2009)


# ---------------------------------------------------------------- balance
def standardized_mean_differences(X: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """Per-feature SMD = (mean_treated - mean_control) / pooled std, plus t-test p-value."""
    t = np.asarray(t)
    rows = []
    for col in X.columns:
        x1, x0 = X.loc[t == 1, col], X.loc[t == 0, col]
        pooled_sd = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
        smd = (x1.mean() - x0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        p_value = stats.ttest_ind(x1, x0, equal_var=False).pvalue
        rows.append(
            {
                "feature": col,
                "mean_treated": x1.mean(),
                "mean_control": x0.mean(),
                "smd": smd,
                "p_value": p_value,
                "balanced": abs(smd) < SMD_THRESHOLD,
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=np.abs, ascending=False, ignore_index=True)


def propensity_auc(X: pd.DataFrame, t: pd.Series, seed: int, folds: int = 5) -> float:
    """Out-of-fold ROC-AUC of a logistic model predicting treatment from features.

    Under true randomization the features carry no information about who was
    treated, so AUC should be ~0.50. Values > ~0.55 signal a selection problem.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, X, t, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(t, scores))


# ---------------------------------------------------------------- effects
@dataclass(frozen=True)
class AteResult:
    """Average treatment effect for a binary outcome (difference in proportions)."""

    n_treated: int
    n_control: int
    rate_treated: float
    rate_control: float
    ate: float  # absolute lift, e.g. 0.045 = +4.5 percentage points
    ci_low: float
    ci_high: float
    relative_lift: float  # ate / rate_control
    p_value: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def estimate_ate(y: pd.Series, t: pd.Series, alpha: float = 0.05) -> AteResult:
    """ATE = P(y=1 | treated) - P(y=1 | control), with a normal-approx CI and z-test.

    Valid as a causal estimate because treatment was randomized.
    """
    y, t = np.asarray(y), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    p1, p0 = y1.mean(), y0.mean()
    n1, n0 = len(y1), len(y0)
    ate = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(ate / se))) if se > 0 else 1.0
    return AteResult(
        n_treated=n1,
        n_control=n0,
        rate_treated=p1,
        rate_control=p0,
        ate=ate,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        relative_lift=ate / p0 if p0 > 0 else np.nan,
        p_value=p_value,
    )


def class_balance(y: pd.Series, t: pd.Series) -> pd.DataFrame:
    """Counts and response rate by group — shows how rare the positive class is."""
    df = pd.DataFrame({"group": np.where(np.asarray(t) == 1, "treated", "control"), "y": y})
    table = df.groupby("group")["y"].agg(n="size", responders="sum", response_rate="mean")
    table.loc["total"] = [len(df), df["y"].sum(), df["y"].mean()]
    return table


def uplift_by_group(df: pd.DataFrame, group_col: str, y_col: str, t_col: str) -> pd.DataFrame:
    """ATE inside each level of a column — a first look at *heterogeneous* effects.

    If uplift differs a lot between groups, a single ATE hides who to target,
    which is exactly what uplift models (CATE) will exploit.
    """
    rows = []
    for level, part in df.groupby(group_col, observed=True):
        res = estimate_ate(part[y_col], part[t_col])
        rows.append({group_col: level, "n": len(part), **res.to_dict()})
    return pd.DataFrame(rows)
