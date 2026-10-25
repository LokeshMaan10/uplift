"""Build every model from the config in one place."""

from __future__ import annotations

from collections.abc import Callable

from upliftlab.config import Config
from upliftlab.models.base import UpliftModel
from upliftlab.models.baselines import HistoricalBuyerModel, RandomModel, ResponseModel
from upliftlab.models.meta import RLearner, SLearner, TLearner, XLearner
from upliftlab.models.other import CausalForest, ClassTransformation, UpliftRandomForest

MODEL_FAMILIES: dict[str, str] = {
    "Random": "Baseline",
    "Response model": "Baseline",
    "Historical buyer": "Baseline",
    "S-learner": "Meta-learner",
    "T-learner": "Meta-learner",
    "X-learner": "Meta-learner",
    "R-learner": "Meta-learner",
    "Class transformation": "Other uplift",
    "Uplift RF (KL)": "Other uplift",
    "Causal forest": "Other uplift",
}

# Pure LightGBM/sklearn models: small, fast, safe to ship in the Streamlit app
LIGHTWEIGHT_MODELS: list[str] = ["S-learner", "T-learner", "X-learner", "R-learner", "Class transformation"]


def model_factories(cfg: Config, seed: int) -> dict[str, Callable[[], UpliftModel]]:
    """Name -> zero-arg function returning a fresh, unfitted model (needed for CV)."""
    lgbm = cfg.lightgbm.to_dict()
    return {
        "Random": lambda: RandomModel(seed),
        "Response model": lambda: ResponseModel(lgbm, seed),
        "Historical buyer": lambda: HistoricalBuyerModel(lgbm, seed),
        "S-learner": lambda: SLearner(lgbm, seed),
        "T-learner": lambda: TLearner(lgbm, seed),
        "X-learner": lambda: XLearner(lgbm, seed),
        "R-learner": lambda: RLearner(lgbm, seed, folds=cfg.split.cv_folds),
        "Class transformation": lambda: ClassTransformation(lgbm, seed),
        "Uplift RF (KL)": lambda: UpliftRandomForest(cfg.uplift_rf.to_dict(), seed),
        "Causal forest": lambda: CausalForest(cfg.causal_forest.to_dict(), lgbm, seed),
    }


def build_models(
    cfg: Config, seed: int, names: list[str] | None = None
) -> dict[str, Callable[[], UpliftModel]]:
    """Factories for the requested model names (all models by default)."""
    factories = model_factories(cfg, seed)
    if names is None:
        return factories
    unknown = set(names) - set(factories)
    if unknown:
        raise KeyError(f"Unknown models: {sorted(unknown)}")
    return {n: factories[n] for n in names}
