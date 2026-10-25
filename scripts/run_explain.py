"""Module 9 — SHAP drivers of persuadability + profile of the 4 predicted segments.

Run after run_pipeline.py:
    python scripts/run_explain.py

Outputs:
    reports/figures/shap_importance_*.png, shap_beeswarm_*.png, segments_validation.png
    reports/tables/shap_importance_*.csv, segments_validation.csv, segments_profile.csv
    app/artifacts/surrogate.joblib (+ fidelity added to metadata.json)
"""

from __future__ import annotations

import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from upliftlab.config import PROJECT_ROOT, load_config
from upliftlab.data import TARGET, TREATMENT, split_xyt, train_test_split_uplift
from upliftlab.eda import estimate_ate
from upliftlab.explain import fit_surrogate, global_importance, shap_values
from upliftlab.plots import apply_style, plot_importance, plot_segment_validation
from upliftlab.segments import SegmentThresholds, assign_segments
from upliftlab.simulate import SEGMENTS
from upliftlab.utils import get_logger, set_seed

log = get_logger("explain")
APP_ARTIFACTS = PROJECT_ROOT / "app" / "artifacts"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def explain_model(name: str, model, X_tr: pd.DataFrame, X_te: pd.DataFrame, cfg, figs, tables):
    """Surrogate + SHAP for one model; returns (surrogate, fidelity R^2, importance)."""
    surrogate, r2_train = fit_surrogate(X_tr, model.predict_uplift(X_tr), cfg.lightgbm.to_dict(), cfg.project.seed)
    pred_te = model.predict_uplift(X_te)
    r2_test = float(1 - np.mean((surrogate.predict(X_te) - pred_te) ** 2) / np.var(pred_te))
    values, _ = shap_values(surrogate, X_te)
    imp = global_importance(values, X_te)
    imp.to_csv(tables / f"shap_importance_{slug(name)}.csv", index=False)
    plot_importance(imp, figs / f"shap_importance_{slug(name)}.png")

    apply_style()
    shap.summary_plot(values, X_te, show=False, max_display=11, plot_size=(7, 4.5))
    plt.title(f"{name}: SHAP on predicted uplift (surrogate R² = {r2_test:.2f})", loc="left", fontsize=11)
    plt.savefig(figs / f"shap_beeswarm_{slug(name)}.png", bbox_inches="tight", dpi=150)
    plt.close("all")
    log.info("%s surrogate fidelity R² train %.3f / test %.3f", name, r2_train, r2_test)
    return surrogate, r2_test, imp


def segment_tables(test: pd.DataFrame, uplift: np.ndarray, baseline: np.ndarray, th: SegmentThresholds):
    """Validation (predicted vs observed uplift) and feature profile per predicted segment."""
    seg = assign_segments(uplift, baseline, th)
    df = test.assign(segment=seg, pred_uplift=uplift, baseline=baseline)
    rows = []
    for s in SEGMENTS:
        part = df[df["segment"] == s]
        if len(part) == 0 or part[TREATMENT].nunique() < 2:
            rows.append({"segment": s, "n": len(part), "share": len(part) / len(df)})
            continue
        ate = estimate_ate(part[TARGET], part[TREATMENT])
        rows.append({"segment": s, "n": len(part), "share": len(part) / len(df),
                     "predicted_uplift": part["pred_uplift"].mean(), "observed_uplift": ate.ate,
                     "ci_low": ate.ci_low, "ci_high": ate.ci_high,
                     "control_rate": ate.rate_control, "treated_rate": ate.rate_treated})
    validation = pd.DataFrame(rows)

    profile_cols = ["recency", "history", "mens", "womens", "newbie",
                    "zip_code_Rural", "zip_code_Surburban", "zip_code_Urban",
                    "channel_Multichannel", "channel_Phone", "channel_Web"]
    profile = df.groupby("segment")[profile_cols].mean().reindex(SEGMENTS).dropna(how="all")
    profile.loc["All customers"] = df[profile_cols].mean()
    return validation, profile


def main() -> None:
    cfg = load_config()
    set_seed(cfg.project.seed)
    figs, tables = cfg.path("figures"), cfg.path("tables")
    meta_path = APP_ARTIFACTS / "metadata.json"
    meta = json.loads(meta_path.read_text())

    h = cfg.data.hillstrom
    df = pd.read_parquet(cfg.path("data_processed") / f"hillstrom_{h.treatment_mode}_{h.target}.parquet")
    train, test = train_test_split_uplift(df, cfg.split.test_size, cfg.project.seed)
    X_tr, _, _ = split_xyt(train)
    X_te, _, _ = split_xyt(test)

    deploy = joblib.load(APP_ARTIFACTS / "deploy_models.joblib")
    best_name = meta["best_model"]
    to_explain = {meta["deploy_model"]: deploy["uplift_model"]}
    if best_name != meta["deploy_model"]:
        to_explain[best_name] = joblib.load(cfg.path("models") / f"{slug(best_name)}.joblib")

    importances = {}
    for name, model in to_explain.items():
        surrogate, r2, imp = explain_model(name, model, X_tr, X_te, cfg, figs, tables)
        importances[name] = imp
        if name == meta["deploy_model"]:
            joblib.dump(surrogate, APP_ARTIFACTS / "surrogate.joblib", compress=3)
            meta["surrogate_r2_test"] = r2

    # Segments use the deployed model so report and app tell the same story
    th = SegmentThresholds(**meta["segment_thresholds"])
    uplift = deploy["uplift_model"].predict_uplift(X_te)
    baseline = deploy["baseline_model"].predict_outcomes(X_te)[0]
    validation, profile = segment_tables(test, uplift, baseline, th)
    validation.to_csv(tables / "segments_validation.csv", index=False)
    profile.to_csv(tables / "segments_profile.csv")
    plot_segment_validation(validation.dropna(subset=["observed_uplift"]), figs / "segments_validation.png")
    meta_path.write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 160)
    for name, imp in importances.items():
        print(f"\n=== SHAP importance — {name}\n", imp.round(4).to_string(index=False))
    print("\n=== Segment validation (test set, deployed model)\n", validation.round(4).to_string(index=False))
    print("\n=== Segment profile (feature means)\n", profile.round(3).to_string())


if __name__ == "__main__":
    main()
