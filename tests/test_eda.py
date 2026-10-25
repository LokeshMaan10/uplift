"""Tests for EDA statistics and data helpers, on synthetic data (no download)."""

import numpy as np
import pandas as pd
import pytest

from upliftlab.data import CONTROL, MENS, TARGET, TREATMENT, WOMENS, select_arms, train_test_split_uplift
from upliftlab.eda import class_balance, estimate_ate, propensity_auc, standardized_mean_differences


@pytest.fixture
def randomized() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Random treatment, true effect +10pp on a 20% base rate."""
    rng = np.random.default_rng(0)
    n = 20_000
    X = pd.DataFrame({"age": rng.normal(40, 10, n), "is_web": rng.integers(0, 2, n)})
    t = pd.Series(rng.integers(0, 2, n))
    y = pd.Series((rng.random(n) < 0.20 + 0.10 * t).astype(int))
    return X, y, t


def test_ate_recovers_true_effect(randomized) -> None:
    _, y, t = randomized
    res = estimate_ate(y, t)
    assert res.ci_low < 0.10 < res.ci_high
    assert res.p_value < 1e-6


def test_ate_exact_on_tiny_example() -> None:
    res = estimate_ate(pd.Series([1, 1, 0, 0, 1, 0]), pd.Series([1, 1, 1, 0, 0, 0]))
    assert res.rate_treated == pytest.approx(2 / 3)
    assert res.rate_control == pytest.approx(1 / 3)
    assert res.ate == pytest.approx(1 / 3)


def test_random_assignment_is_balanced(randomized) -> None:
    X, _, t = randomized
    assert standardized_mean_differences(X, t)["balanced"].all()
    assert abs(propensity_auc(X, t, seed=0) - 0.5) < 0.03


def test_imbalance_is_detected() -> None:
    rng = np.random.default_rng(1)
    age = rng.normal(40, 10, 5000)
    t = pd.Series((age > 40).astype(int))  # older people got the offer -> not random
    X = pd.DataFrame({"age": age})
    assert not standardized_mean_differences(X, t)["balanced"].all()
    assert propensity_auc(X, t, seed=0) > 0.9


def test_class_balance_totals(randomized) -> None:
    _, y, t = randomized
    cb = class_balance(y, t)
    assert cb.loc["total", "n"] == len(y)


def test_select_arms_womens_drops_mens() -> None:
    raw = pd.DataFrame({"arm": [MENS, WOMENS, CONTROL, CONTROL], TARGET: [1, 0, 0, 1]})
    out = select_arms(raw, "womens")
    assert MENS not in set(out["arm"])
    assert out[TREATMENT].tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        select_arms(raw, "nope")


def test_stratified_split_keeps_rates(randomized) -> None:
    X, y, t = randomized
    df = X.assign(**{TREATMENT: t, TARGET: y})
    train, test = train_test_split_uplift(df, test_size=0.3, seed=0)
    assert len(test) == pytest.approx(0.3 * len(df), abs=1)
    assert train[TREATMENT].mean() == pytest.approx(test[TREATMENT].mean(), abs=0.005)
    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=0.005)
