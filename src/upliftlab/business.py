"""Module 8 — turn uplift scores into money.

For "target the top p% by model score":
    uplift_p        = observed treated rate - control rate within that top p%
                      (measured on the held-out test set, so it is causal + honest)
    n_targeted      = population * p
    incremental     = uplift_p * n_targeted          extra responses caused by the offer
    profit          = incremental * value_per_response - n_targeted * cost_per_offer
    ROI             = profit / campaign cost

Random targeting of p% earns the average effect (ATE) on every contact, so its
profit is a straight line; "mail everyone" is the p = 100% point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def uplift_at_percents(y: ArrayLike, score: ArrayLike, t: ArrayLike, percents: list[float]) -> pd.DataFrame:
    """Observed uplift among the top p% for each p (sorted by score, descending)."""
    y, score, t = np.asarray(y), np.asarray(score), np.asarray(t)
    order = np.argsort(-score, kind="stable")
    y_s, t_s = y[order], t[order]
    rows = []
    for p in percents:
        k = max(1, int(round(len(y) * p / 100)))
        yk, tk = y_s[:k], t_s[:k]
        rate_t = yk[tk == 1].mean() if (tk == 1).any() else np.nan
        rate_c = yk[tk == 0].mean() if (tk == 0).any() else np.nan
        rows.append({"percent": p, "uplift": rate_t - rate_c})
    return pd.DataFrame(rows)


def profit_table(uplift_df: pd.DataFrame, cost: float, value: float, population: int) -> pd.DataFrame:
    """Add campaign economics to an uplift-by-percent table."""
    df = uplift_df.copy()
    df["n_targeted"] = (population * df["percent"] / 100).round().astype(int)
    df["incremental_responses"] = df["uplift"] * df["n_targeted"]
    df["revenue"] = df["incremental_responses"] * value
    df["cost"] = df["n_targeted"] * cost
    df["profit"] = df["revenue"] - df["cost"]
    df["roi"] = np.where(df["cost"] > 0, df["profit"] / df["cost"], np.nan)
    return df


def random_profit_table(ate: float, percents: list[float], cost: float, value: float, population: int) -> pd.DataFrame:
    """Expected economics of random targeting: every contact earns the ATE."""
    base = pd.DataFrame({"percent": percents, "uplift": ate})
    return profit_table(base, cost, value, population)


def optimal_cutoff(table: pd.DataFrame) -> pd.Series:
    """Row with maximum profit (the best % of customers to target).

    If every option loses money the best decision is to not run the campaign:
    a 0% row with zero cost and zero profit is returned.
    """
    best = table.loc[table["profit"].idxmax()]
    if best["profit"] >= 0:
        return best
    none = pd.Series(0.0, index=table.columns)
    none["roi"] = np.nan
    return none


def what_if_costs(
    y: ArrayLike,
    score: ArrayLike,
    t: ArrayLike,
    costs: list[float],
    value: float,
    population: int,
    percents: list[float],
) -> pd.DataFrame:
    """For each offer cost: optimal % to target, its profit, and 'mail everyone' profit."""
    upl = uplift_at_percents(y, score, t, percents)
    ate = float(upl.loc[upl["percent"] == 100, "uplift"].iloc[0]) if 100 in percents else np.nan
    rows = []
    for c in costs:
        tbl = profit_table(upl, c, value, population)
        best = optimal_cutoff(tbl)
        everyone = population * ate * value - population * c
        rows.append({
            "cost_per_offer": c,
            "breakeven_uplift": c / value,
            "optimal_percent": best["percent"],
            "optimal_profit": best["profit"],
            "mail_everyone_profit": everyone,
            "gain_vs_everyone": best["profit"] - everyone,
        })
    return pd.DataFrame(rows)


def uplift_by_bins(y: ArrayLike, score: ArrayLike, t: ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Observed uplift per score bin (bin 1 = highest scores). Dependency-free decile table."""
    y, score, t = np.asarray(y), np.asarray(score), np.asarray(t)
    order = np.argsort(-score, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), start=1):
        yb, tb = y[idx], t[idx]
        p1, p0 = yb[tb == 1].mean(), yb[tb == 0].mean()
        n1, n0 = (tb == 1).sum(), (tb == 0).sum()
        se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        rows.append({"bin": i, "n": len(idx), "uplift": p1 - p0, "ci_low": p1 - p0 - 1.96 * se,
                     "ci_high": p1 - p0 + 1.96 * se, "mean_score": score[idx].mean()})
    return pd.DataFrame(rows)
