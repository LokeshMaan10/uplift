"""Every model fits, returns one score per row, and the uplift models track the true uplift."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from upliftlab.data import TARGET, TREATMENT
from upliftlab.models import LIGHTWEIGHT_MODELS, MODEL_FAMILIES, build_models
from upliftlab.models.meta import SLearner, TLearner
from upliftlab.models.other import ClassTransformation
from upliftlab.simulate import FEATURES


@pytest.mark.parametrize("name", LIGHTWEIGHT_MODELS + ["Response model", "Random"])
def test_model_fits_and_scores_every_row(cfg, sim_split, name) -> None:
    train, test = sim_split
    model = build_models(cfg, 0, [name])[name]().fit(train[FEATURES], train[TARGET], train[TREATMENT])
    score = model.predict_uplift(test[FEATURES])
    assert score.shape == (len(test),)
    assert np.isfinite(score).all()


@pytest.mark.parametrize("name", LIGHTWEIGHT_MODELS)
def test_uplift_models_recover_true_uplift(cfg, sim_split, name) -> None:
    train, test = sim_split
    model = build_models(cfg, 0, [name])[name]().fit(train[FEATURES], train[TARGET], train[TREATMENT])
    score = model.predict_uplift(test[FEATURES])
    assert spearmanr(score, test["true_cate"])[0] > 0.3
    assert abs(score.mean() - test["true_cate"].mean()) < 0.05  # average ≈ true ATE


def test_outcome_models_return_probabilities(cfg, sim_split) -> None:
    train, test = sim_split
    for cls in (SLearner, TLearner):
        p0, p1 = cls(cfg.lightgbm.to_dict()).fit(train[FEATURES], train[TARGET], train[TREATMENT]) \
            .predict_outcomes(test[FEATURES])
        assert ((p0 >= 0) & (p0 <= 1) & (p1 >= 0) & (p1 <= 1)).all()


def test_class_transformation_handles_unbalanced_treatment(cfg, sim_split) -> None:
    """With 85% treated (like Criteo) reweighting keeps the average uplift unbiased."""
    train, test = sim_split
    rng = np.random.default_rng(0)
    keep = (train[TREATMENT] == 1) | (rng.random(len(train)) < 0.18)  # drop most controls
    sub = train[keep]
    assert sub[TREATMENT].mean() > 0.8
    score = ClassTransformation(cfg.lightgbm.to_dict()).fit(sub[FEATURES], sub[TARGET], sub[TREATMENT]) \
        .predict_uplift(test[FEATURES])
    assert abs(score.mean() - test["true_cate"].mean()) < 0.06


def test_registry_covers_all_families_and_rejects_unknown(cfg) -> None:
    assert set(build_models(cfg, 0)) == set(MODEL_FAMILIES)
    with pytest.raises(KeyError):
        build_models(cfg, 0, ["Magic model"])
