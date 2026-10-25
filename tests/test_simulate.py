"""Tests for the synthetic customer simulator."""

import numpy as np
import pytest

from upliftlab.simulate import SEGMENTS, SLEEPING_DOG, simulate_customers


@pytest.fixture(scope="module")
def sim():
    return simulate_customers(n=20_000, seed=0)


def test_all_four_segments_present(sim) -> None:
    assert set(sim["segment"]) == set(SEGMENTS)


def test_segments_match_potential_outcomes(sim) -> None:
    dogs = sim[sim["segment"] == SLEEPING_DOG]
    assert (dogs["y0"] == 1).all() and (dogs["y1"] == 0).all()


def test_observed_outcome_is_the_treated_or_control_one(sim) -> None:
    expected = np.where(sim["treatment"] == 1, sim["y1"], sim["y0"])
    assert (sim["target"] == expected).all()


def test_diff_in_means_recovers_true_ate(sim) -> None:
    t = sim["treatment"]
    est = sim.loc[t == 1, "target"].mean() - sim.loc[t == 0, "target"].mean()
    assert est == pytest.approx(sim["true_cate"].mean(), abs=0.015)


def test_same_seed_is_reproducible() -> None:
    a, b = simulate_customers(500, seed=7), simulate_customers(500, seed=7)
    assert a.equals(b)
