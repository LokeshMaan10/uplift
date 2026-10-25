"""Modules 4-8 + 11 — train every model, evaluate, simulate ROI, log to MLflow.

Run from the project root (venv active):
    python scripts/run_pipeline.py            # all models (~8 min on a laptop)
    python scripts/run_pipeline.py --fast     # skip the slow uplift random forest

Outputs:
    reports/tables/  model_comparison.csv, cv_folds.csv, deciles.csv, curves.csv,
                     sim_recovery.csv, roi_by_percent.csv, roi_optimal.csv, roi_what_if.csv
    reports/figures/ qini_*.png, uplift_curves.png, model_ci.png, deciles_best.png,
                     roi_profit_curves.png, roi_what_if.png
    models/*.joblib  every fitted model
    app/artifacts/   small files the Streamlit app needs
    mlruns/          one MLflow run per model
"""

from __future__ import annotations

import argparse
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from upliftlab.business import (
    optimal_cutoff,
    profit_table,
    random_profit_table,
    uplift_at_percents,
    what_if_costs,
)
from upliftlab.config import PROJECT_ROOT, Config, load_config
from upliftlab.data import TARGET, TREATMENT, load_hillstrom, split_xyt, train_test_split_uplift
from upliftlab.eda import estimate_ate
from upliftlab.evaluation import cross_validate, summarize_cv
from upliftlab.metrics import bootstrap_ci, curve_points, decile_table, uplift_metrics
from upliftlab.models import LIGHTWEIGHT_MODELS, MODEL_FAMILIES, build_models
from upliftlab.plots import (
    plot_curves,
    plot_deciles,
    plot_metric_ci,
    plot_profit_curves,
    plot_what_if,
)
from upliftlab.segments import fit_thresholds
from upliftlab.simulate import FEATURES as SIM_FEATURES
from upliftlab.simulate import simulate_customers
from upliftlab.tracking import log_run, setup_mlflow
from upliftlab.utils import get_logger, set_seed

log = get_logger("pipeline")
APP_ARTIFACTS = PROJECT_ROOT / "app" / "artifacts"
WHAT_IF_COSTS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_data(cfg: Config) -> pd.DataFrame:
    h = cfg.data.hillstrom
    path = cfg.path("data_processed") / f"hillstrom_{h.treatment_mode}_{h.target}.parquet"
    return pd.read_parquet(path) if path.exists() else load_hillstrom(cfg)


# ---------------------------------------------------------------- Modules 4-7, 11
def train_and_evaluate(cfg: Config, train: pd.DataFrame, test: pd.DataFrame, names: list[str]) -> dict:
    """CV on train, refit on all train, score test, bootstrap CI, log to MLflow."""
    seed, ks = cfg.project.seed, cfg.evaluation.top_k_percents
    X_tr, y_tr, t_tr = split_xyt(train)
    X_te, y_te, t_te = split_xyt(test)
    setup_mlflow(cfg)
    models_dir = cfg.path("models")

    rows, cv_all, curves, deciles, fitted, scores = [], [], [], [], {}, {}
    for name, factory in build_models(cfg, seed, names).items():
        start = time.time()
        cv = cross_validate(factory, X_tr, y_tr, t_tr, cfg.split.cv_folds, seed, ks)
        model = factory().fit(X_tr, y_tr, t_tr)
        score = model.predict_uplift(X_te)
        test_m = uplift_metrics(y_te, score, t_te, ks)
        _, lo, hi = bootstrap_ci(y_te, score, t_te, n_iter=cfg.evaluation.bootstrap_iterations, seed=seed)
        elapsed = time.time() - start

        row = {"model": name, "family": MODEL_FAMILIES[name], **summarize_cv(cv),
               **{f"test_{k}": v for k, v in test_m.items()},
               "test_qini_ci_low": lo, "test_qini_ci_high": hi, "fit_seconds": elapsed}
        rows.append(row)
        cv_all.append(cv.assign(model=name))
        for kind in ("qini", "uplift"):
            curves.append(curve_points(y_te, score, t_te, kind).assign(model=name, kind=kind))
        deciles.append(decile_table(y_te, score, t_te, cfg.evaluation.n_bins).assign(model=name))
        fitted[name], scores[name] = model, score

        model_path = models_dir / f"{slug(name)}.joblib"
        joblib.dump(model, model_path)
        log_run(
            run_name=name,
            params={"model": name, "family": MODEL_FAMILIES[name], "seed": seed,
                    "data": cfg.data.hillstrom.to_dict(), "split": cfg.split.to_dict(),
                    "lightgbm": cfg.lightgbm.to_dict() if name != "Random" else {},
                    **({"uplift_rf": cfg.uplift_rf.to_dict()} if "RF" in name else {}),
                    **({"causal_forest": cfg.causal_forest.to_dict()} if "forest" in name else {})},
            metrics={k: v for k, v in row.items() if isinstance(v, float)},
            artifacts=[model_path],
            tags={"dataset": "hillstrom", "module": "7"},
        )
        log.info("%-22s cv qini %.4f ± %.4f | test qini %.4f [%.4f, %.4f] | %.0fs",
                 name, row["cv_qini_coef_mean"], row["cv_qini_coef_std"], test_m["qini_coef"], lo, hi, elapsed)

    comparison = pd.DataFrame(rows).sort_values("test_qini_coef", ascending=False, ignore_index=True)
    return {"comparison": comparison, "cv": pd.concat(cv_all), "curves": pd.concat(curves),
            "deciles": pd.concat(deciles), "fitted": fitted, "scores": scores}


def simulation_recovery(cfg: Config, names: list[str]) -> pd.DataFrame:
    """Sanity check with known truth: does each model recover the TRUE uplift?"""
    sim = simulate_customers(20_000, seed=cfg.project.seed)
    tr, te = train_test_split_uplift(sim[SIM_FEATURES + [TREATMENT, TARGET, "true_cate"]], 0.3, cfg.project.seed)
    oracle = uplift_metrics(te[TARGET], te["true_cate"], te[TREATMENT], [30])["qini_coef"]
    rows = []
    for name, factory in build_models(cfg, cfg.project.seed, names).items():
        m = factory().fit(tr[SIM_FEATURES], tr[TARGET], tr[TREATMENT])
        u = m.predict_uplift(te[SIM_FEATURES])
        is_uplift = MODEL_FAMILIES[name] != "Baseline"
        rows.append({
            "model": name,
            "spearman_vs_true_uplift": float(spearmanr(u, te["true_cate"])[0]),
            "rmse_vs_true_uplift": float(np.sqrt(np.mean((u - te["true_cate"]) ** 2))) if is_uplift else np.nan,
            "qini_coef": uplift_metrics(te[TARGET], u, te[TREATMENT], [30])["qini_coef"],
            "oracle_qini_coef": oracle,
        })
    return pd.DataFrame(rows).sort_values("spearman_vs_true_uplift", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- Module 8
def business_simulation(cfg: Config, test: pd.DataFrame, scores: dict[str, np.ndarray], strategies: list[str]):
    b = cfg.business
    percents = list(range(5, 101, 5))
    y, t = test[TARGET], test[TREATMENT]
    ate = estimate_ate(y, t).ate

    tables, long_rows, optimal_rows = {}, [], []
    for name in strategies:
        tbl = profit_table(uplift_at_percents(y, scores[name], t, percents),
                           b.cost_per_offer, b.value_per_response, b.population_size)
        tables[name] = tbl
        long_rows.append(tbl.assign(strategy=name))
        best = optimal_cutoff(tbl)
        optimal_rows.append({"strategy": name, "optimal_percent": best["percent"], "profit": best["profit"],
                             "roi": best["roi"], "incremental_responses": best["incremental_responses"]})
    rand = random_profit_table(ate, percents, b.cost_per_offer, b.value_per_response, b.population_size)
    tables["Random targeting"] = rand
    long_rows.append(rand.assign(strategy="Random targeting"))
    everyone = rand.iloc[-1]
    optimal_rows.append({"strategy": "Mail everyone", "optimal_percent": 100, "profit": everyone["profit"],
                         "roi": everyone["roi"], "incremental_responses": everyone["incremental_responses"]})

    optimal = pd.DataFrame(optimal_rows).sort_values("profit", ascending=False, ignore_index=True)
    what_if = what_if_costs(y, scores[strategies[0]], t, WHAT_IF_COSTS, b.value_per_response,
                            b.population_size, percents)
    return tables, pd.concat(long_rows, ignore_index=True), optimal, what_if, ate


# ---------------------------------------------------------------- app artifacts
def export_app_artifacts(cfg: Config, train: pd.DataFrame, test: pd.DataFrame, res: dict,
                         deploy_name: str, ate: float) -> None:
    """Small files the Streamlit app loads (kept in git so the hosted app works)."""
    APP_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    X_tr, _, _ = split_xyt(train)
    X_te, _, _ = split_xyt(test)
    deploy, tlearner = res["fitted"][deploy_name], res["fitted"]["T-learner"]

    seg_cfg = cfg.segments
    th = fit_thresholds(deploy.predict_uplift(X_tr), tlearner.predict_outcomes(X_tr)[0],
                        seg_cfg.persuadable_quantile, seg_cfg.sleeping_dog_uplift, seg_cfg.sure_thing_quantile)

    scores = test.copy()
    for name, s in res["scores"].items():
        scores[f"score__{name}"] = s
    scores["baseline_p0"] = tlearner.predict_outcomes(X_te)[0]
    scores.to_csv(APP_ARTIFACTS / "test_scores.csv", index=False)

    joblib.dump({"uplift_model": deploy, "baseline_model": tlearner, "model_name": deploy_name},
                APP_ARTIFACTS / "deploy_models.joblib", compress=3)
    res["comparison"].to_csv(APP_ARTIFACTS / "model_comparison.csv", index=False)
    res["curves"].to_csv(APP_ARTIFACTS / "curves.csv", index=False)
    meta = {
        "deploy_model": deploy_name,
        "best_model": res["comparison"].query("family != 'Baseline'").iloc[0]["model"],
        "features": list(X_tr.columns),
        "numeric_ranges": {c: [float(X_tr[c].min()), float(X_tr[c].max()), float(X_tr[c].median())]
                           for c in ["recency", "history"]},
        "ate": ate,
        "segment_thresholds": th.to_dict(),
        "business_defaults": cfg.business.to_dict(),
        "data": cfg.data.hillstrom.to_dict(),
        "n_train": len(train), "n_test": len(test),
    }
    (APP_ARTIFACTS / "metadata.json").write_text(json.dumps(meta, indent=2))


def render_figures(cfg: Config) -> None:
    """Draw every Module 7-8 figure from the saved CSV tables (no re-training needed)."""
    figs, tables_dir = cfg.path("figures"), cfg.path("tables")
    comp = pd.read_csv(tables_dir / "model_comparison.csv")
    curves = pd.read_csv(tables_dir / "curves.csv")
    deciles = pd.read_csv(tables_dir / "deciles.csv")
    roi_long = pd.read_csv(tables_dir / "roi_by_percent.csv")
    what_if = pd.read_csv(tables_dir / "roi_what_if.csv")
    test_ate = float(roi_long.query("strategy == 'Random targeting'")["uplift"].iloc[0])

    uplift_models = comp.query("family != 'Baseline'")["model"].tolist()
    best = uplift_models[0]
    qini, upl = curves.query("kind == 'qini'"), curves.query("kind == 'uplift'")
    pick = lambda c, ms: {m: c[c["model"] == m] for m in ms}  # noqa: E731
    ylab = "Incremental responders (% of customers)"

    plot_curves(pick(qini, ["Response model", "Historical buyer", best]),
                "Qini: response models vs best uplift model (test set)", ylab, figs / "qini_baselines_vs_best.png")
    plot_curves(pick(qini, uplift_models), "Qini curves: all uplift models (test set)", ylab,
                figs / "qini_uplift_models.png")
    plot_curves(pick(upl, uplift_models), "Uplift curves: all uplift models (test set)", ylab,
                figs / "uplift_curves.png")
    plot_metric_ci(comp, "test_qini_coef", "test_qini_ci_low", "test_qini_ci_high",
                   "Qini coefficient on the test set", figs / "model_ci.png")
    plot_deciles(deciles[deciles["model"] == best], best, test_ate, figs / "deciles_best.png")
    roi_tables = {s: g.reset_index(drop=True) for s, g in roi_long.groupby("strategy", sort=False)}
    plot_profit_curves(roi_tables, cfg.business.currency, figs / "roi_profit_curves.png")
    plot_what_if(what_if, cfg.business.currency, figs / "roi_what_if.png")
    log.info("Figures written to %s", figs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip Uplift RF (slowest model)")
    parser.add_argument("--figures-only", action="store_true", help="redraw figures from saved tables")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.project.seed)
    if args.figures_only:
        render_figures(cfg)
        return
    tables_dir = cfg.path("tables")
    names = [n for n in MODEL_FAMILIES if not (args.fast and n == "Uplift RF (KL)")]

    df = load_data(cfg)
    train, test = train_test_split_uplift(df, cfg.split.test_size, cfg.project.seed)
    log.info("Train %d rows | test %d rows | %d models", len(train), len(test), len(names))

    # Modules 4-7 + 11
    res = train_and_evaluate(cfg, train, test, names)
    comp = res["comparison"]
    comp.to_csv(tables_dir / "model_comparison.csv", index=False)
    res["cv"].to_csv(tables_dir / "cv_folds.csv", index=False)
    res["curves"].to_csv(tables_dir / "curves.csv", index=False)
    res["deciles"].to_csv(tables_dir / "deciles.csv", index=False)

    uplift_models = comp.query("family != 'Baseline'")["model"].tolist()
    best = uplift_models[0]
    deploy = next(m for m in uplift_models if m in LIGHTWEIGHT_MODELS)

    log.info("Simulation recovery check...")
    sim_rec = simulation_recovery(cfg, names)
    sim_rec.to_csv(tables_dir / "sim_recovery.csv", index=False)

    # Module 8
    strategies = list(dict.fromkeys([best, deploy, "Response model", "Historical buyer"]))
    roi_tables, roi_long, roi_opt, what_if, ate = business_simulation(cfg, test, res["scores"], strategies)
    roi_long.to_csv(tables_dir / "roi_by_percent.csv", index=False)
    roi_opt.to_csv(tables_dir / "roi_optimal.csv", index=False)
    what_if.to_csv(tables_dir / "roi_what_if.csv", index=False)

    export_app_artifacts(cfg, train, test, res, deploy, ate)
    render_figures(cfg)

    pd.set_option("display.width", 160)
    cols = ["model", "family", "cv_qini_coef_mean", "cv_qini_coef_std", "test_qini_coef",
            "test_qini_ci_low", "test_qini_ci_high", "test_auuc", "test_uplift@10%", "test_uplift@30%"]
    print("\n=== Model comparison (sorted by test Qini coefficient)\n", comp[cols].round(4).to_string(index=False))
    print("\n=== Simulation: recovery of the TRUE uplift\n", sim_rec.round(3).to_string(index=False))
    print(f"\n=== ROI @ cost {cur} {cfg.business.cost_per_offer}, value {cur} {cfg.business.value_per_response}, "
          f"population {cfg.business.population_size:,}\n", roi_opt.round(2).to_string(index=False))
    print("\n=== What-if: offer cost (best model)\n", what_if.round(2).to_string(index=False))
    print(f"\nBest uplift model: {best} | deployed in app: {deploy}")


if __name__ == "__main__":
    main()
