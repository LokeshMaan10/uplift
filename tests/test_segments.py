"""Segment rules and the explainability helpers."""

import numpy as np
import pandas as pd
import pytest

from upliftlab.explain import fit_surrogate, global_importance, shap_values
from upliftlab.segments import SegmentThresholds, assign_segments, fit_thresholds


def test_assign_segments_rules() -> None:
    th = SegmentThresholds(persuadable_uplift=0.05, sleeping_dog_uplift=0.0, sure_thing_baseline=0.2)
    uplift = np.array([0.10, -0.02, 0.01, 0.01])
    baseline = np.array([0.05, 0.50, 0.40, 0.05])
    assert list(assign_segments(uplift, baseline, th)) == ["Persuadable", "Sleeping Dog", "Sure Thing", "Lost Cause"]


def test_fit_thresholds_uses_quantiles() -> None:
    th = fit_thresholds(np.linspace(0, 1, 101), np.linspace(0, 1, 101), 0.7, 0.0, 0.5)
    assert th.persuadable_uplift == pytest.approx(0.7)
    assert th.sure_thing_baseline == pytest.approx(0.5)


def test_shap_finds_the_driver(cfg) -> None:
    """Uplift that depends only on 'a' -> SHAP ranks 'a' first with positive direction."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(3000), "b": rng.random(3000)})
    uplift = 0.1 * X["a"].to_numpy()
    surrogate, r2 = fit_surrogate(X, uplift, cfg.lightgbm.to_dict(), 0)
    values, _ = shap_values(surrogate, X)
    imp = global_importance(values, X)
    assert r2 > 0.9
    assert imp.iloc[0]["feature"] == "a" and imp.iloc[0]["direction"] > 0.9
