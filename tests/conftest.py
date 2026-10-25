"""Shared fixtures: a simulated dataset with known true uplift."""

import pandas as pd
import pytest

from upliftlab.config import load_config
from upliftlab.data import TARGET, TREATMENT, train_test_split_uplift
from upliftlab.simulate import FEATURES, simulate_customers


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sim_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = simulate_customers(n=12_000, seed=3)
    return train_test_split_uplift(sim[FEATURES + [TREATMENT, TARGET, "true_cate"]], 0.3, seed=0)
