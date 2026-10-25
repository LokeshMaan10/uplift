"""Pick the shared LightGBM settings by cross-validated Qini on the TRAINING set only.

Run:  python scripts/tune_lightgbm.py
The test set is never touched, so the final test metrics stay honest.
"""

from __future__ import annotations

import pandas as pd

from upliftlab.config import load_config
from upliftlab.data import split_xyt, train_test_split_uplift
from upliftlab.evaluation import cross_validate
from upliftlab.models.meta import RLearner, SLearner, TLearner, XLearner
from upliftlab.models.other import ClassTransformation

CANDIDATES: dict[str, dict] = {
    "flexible (300 trees, 31 leaves)": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50),
    "regularized (200 trees, 15 leaves)": dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200),
    "strong (150 trees, 7 leaves)": dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=400),
}
LEARNERS = {"S-learner": SLearner, "T-learner": TLearner, "X-learner": XLearner,
            "R-learner": RLearner, "Class transformation": ClassTransformation}


def main() -> None:
    cfg = load_config()
    seed = cfg.project.seed
    h = cfg.data.hillstrom
    df = pd.read_parquet(cfg.path("data_processed") / f"hillstrom_{h.treatment_mode}_{h.target}.parquet")
    train, _ = train_test_split_uplift(df, cfg.split.test_size, seed)
    X, y, t = split_xyt(train)
    base = cfg.lightgbm.to_dict()

    rows = []
    for label, overrides in CANDIDATES.items():
        params = {**base, **overrides}
        for name, cls in LEARNERS.items():
            cv = cross_validate(lambda: cls(params, seed), X, y, t, cfg.split.cv_folds, seed, [30])
            rows.append({"params": label, "model": name, "cv_qini": cv["qini_coef"].mean()})
    res = pd.DataFrame(rows).pivot(index="model", columns="params", values="cv_qini")
    res.loc["MEAN"] = res.mean()
    print(res.round(4).to_string())
    res.to_csv(cfg.path("tables") / "lightgbm_tuning_cv.csv")


if __name__ == "__main__":
    main()
