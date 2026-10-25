"""ROI arithmetic and decision rules."""

import numpy as np
import pandas as pd
import pytest

from upliftlab.business import (
    optimal_cutoff,
    profit_table,
    random_profit_table,
    uplift_at_percents,
    uplift_by_bins,
    what_if_costs,
)
from upliftlab.data import TARGET, TREATMENT


def test_profit_arithmetic() -> None:
    tbl = profit_table(pd.DataFrame({"percent": [10], "uplift": [0.05]}), cost=1.0, value=30.0, population=1000)
    r = tbl.iloc[0]
    assert r["n_targeted"] == 100
    assert r["incremental_responses"] == pytest.approx(5)
    assert r["profit"] == pytest.approx(5 * 30 - 100)  # 50
    assert r["roi"] == pytest.approx(0.5)


def test_random_targeting_is_linear_in_percent() -> None:
    tbl = random_profit_table(0.04, [10, 50, 100], cost=1, value=30, population=10_000)
    assert tbl["profit"].iloc[2] == pytest.approx(10 * tbl["profit"].iloc[0])


def test_optimal_cutoff_picks_max_profit() -> None:
    tbl = profit_table(pd.DataFrame({"percent": [10, 20, 30], "uplift": [0.2, 0.12, 0.05]}), 1, 30, 1000)
    assert optimal_cutoff(tbl)["percent"] == 20


def test_optimal_cutoff_says_do_nothing_when_all_lose() -> None:
    tbl = profit_table(pd.DataFrame({"percent": [10, 50], "uplift": [0.01, 0.005]}), 5, 30, 1000)
    best = optimal_cutoff(tbl)
    assert best["percent"] == 0 and best["profit"] == 0


def test_uplift_at_100_percent_equals_ate(sim_split) -> None:
    _, test = sim_split
    y, t = test[TARGET], test[TREATMENT]
    u = uplift_at_percents(y, test["true_cate"], t, [100])["uplift"].iloc[0]
    assert u == pytest.approx(y[t == 1].mean() - y[t == 0].mean())


def test_good_ranking_targets_fewer_as_cost_rises(sim_split) -> None:
    _, test = sim_split
    w = what_if_costs(test[TARGET], test["true_cate"], test[TREATMENT], [0.5, 3.0], 30, 10_000,
                      list(range(5, 101, 5)))
    assert w["optimal_percent"].iloc[1] <= w["optimal_percent"].iloc[0]
    assert (w["gain_vs_everyone"] >= -1e-9).all()


def test_uplift_by_bins_partitions_everyone(sim_split) -> None:
    _, test = sim_split
    b = uplift_by_bins(test[TARGET], test["true_cate"], test[TREATMENT], 10)
    assert b["n"].sum() == len(test)
    assert np.all(np.diff(b["mean_score"]) <= 0)
