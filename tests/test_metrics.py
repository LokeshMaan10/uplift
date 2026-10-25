"""Uplift metrics behave as theory says: perfect > model > random, CIs bracket the estimate."""

import numpy as np
import pytest

from upliftlab.data import TARGET, TREATMENT
from upliftlab.metrics import bootstrap_ci, curve_points, decile_table, uplift_metrics


def test_true_uplift_beats_random_ranking(sim_split) -> None:
    _, test = sim_split
    y, t = test[TARGET], test[TREATMENT]
    oracle = uplift_metrics(y, test["true_cate"], t, [10, 30])
    random = uplift_metrics(y, np.random.default_rng(0).random(len(test)), t, [10, 30])
    assert oracle["qini_coef"] > random["qini_coef"] + 0.05
    assert oracle["uplift@10%"] > random["uplift@10%"]


def test_reversed_ranking_is_worse_than_random(sim_split) -> None:
    _, test = sim_split
    m = uplift_metrics(test[TARGET], -test["true_cate"], test[TREATMENT], [30])
    assert m["qini_coef"] < 0


def test_bootstrap_ci_brackets_point_estimate(sim_split) -> None:
    _, test = sim_split
    point, lo, hi = bootstrap_ci(test[TARGET], test["true_cate"], test[TREATMENT], n_iter=100, seed=0)
    assert lo < point < hi


def test_curve_starts_at_zero_and_ends_at_total_effect(sim_split) -> None:
    _, test = sim_split
    c = curve_points(test[TARGET], test["true_cate"], test[TREATMENT], "uplift")
    assert c["frac_targeted"].iloc[0] == 0 and c["frac_targeted"].iloc[-1] == 1
    assert c["incremental_per_customer"].iloc[0] == pytest.approx(0, abs=1e-9)
    t, y = test[TREATMENT], test[TARGET]
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert c["incremental_per_customer"].iloc[-1] == pytest.approx(ate, abs=0.005)


def test_decile_table_has_ten_rows_top_first(sim_split) -> None:
    _, test = sim_split
    d = decile_table(test[TARGET], test["true_cate"], test[TREATMENT])
    assert len(d) == 10
    assert d["uplift"].iloc[0] > d["uplift"].iloc[-1]
