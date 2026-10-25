"""Module 10 — do the conclusions hold up?

Checks:
    1. Seed stability  — re-split + re-train Hillstrom with 5 seeds
    2. Other treatment — same pipeline on the MEN's email arm
    3. Scale + domain  — Criteo uplift (500k-row stratified subsample, 85% treated)

Run after run_pipeline.py (Criteo downloads ~300 MB the first time):
    python scripts/run_robustness.py
    python scripts/run_robustness.py --skip-criteo
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd

from upliftlab.config import load_config
from upliftlab.data import (
    TARGET,
    TREATMENT,
    encode_hillstrom,
    fetch_hillstrom_raw,
    load_criteo,
    select_arms,
    split_xyt,
    train_test_split_uplift,
)
from upliftlab.eda import estimate_ate
from upliftlab.metrics import bootstrap_ci, uplift_metrics
from upliftlab.models import MODEL_FAMILIES, build_models
from upliftlab.plots import BLUE, INK_MUTED, NEUTRAL, apply_style
from upliftlab.tracking import log_run, setup_mlflow
from upliftlab.utils import get_logger, set_seed

log = get_logger("robustness")
HILLSTROM_MODELS = ["Random", "Response model", "S-learner", "T-learner", "X-learner",
                    "R-learner", "Class transformation", "Causal forest"]


def evaluate_split(cfg, df: pd.DataFrame, seed: int, names: list[str], dataset: str,
                   ci_iter: int = 0) -> pd.DataFrame:
    """Train each model on one split and return test metrics (optionally with bootstrap CI)."""
    train, test = train_test_split_uplift(df, cfg.split.test_size, seed)
    X_tr, y_tr, t_tr = split_xyt(train)
    X_te, y_te, t_te = split_xyt(test)
    rows = []
    for name, factory in build_models(cfg, seed, names).items():
        start = time.time()
        score = factory().fit(X_tr, y_tr, t_tr).predict_uplift(X_te)
        m = uplift_metrics(y_te, score, t_te, [10, 30])
        if ci_iter:
            _, m["qini_ci_low"], m["qini_ci_high"] = bootstrap_ci(y_te, score, t_te, n_iter=ci_iter, seed=seed)
        m["fit_seconds"] = time.time() - start
        rows.append({"dataset": dataset, "seed": seed, "model": name, "family": MODEL_FAMILIES[name], **m})
        log.info("%-10s seed %d  %-22s qini %.4f  (%.0fs)", dataset, seed, name, m["qini_coef"], m["fit_seconds"])
    return pd.DataFrame(rows)


def seed_stability(cfg) -> pd.DataFrame:
    h = cfg.data.hillstrom
    df = pd.read_parquet(cfg.path("data_processed") / f"hillstrom_{h.treatment_mode}_{h.target}.parquet")
    return pd.concat([evaluate_split(cfg, df, s, HILLSTROM_MODELS, "hillstrom") for s in cfg.robustness.seeds])


def plot_seed_stability(res: pd.DataFrame, path) -> None:
    apply_style()
    summary = res.groupby("model")["qini_coef"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(summary) + 1.2))
    for i, model in enumerate(summary.index):
        vals = res.loc[res["model"] == model, "qini_coef"]
        color = NEUTRAL if MODEL_FAMILIES[model] == "Baseline" else BLUE
        ax.scatter(vals, [i] * len(vals), color=color, alpha=0.55, s=30, edgecolor="white", linewidth=1)
        ax.scatter(summary.loc[model, "mean"], i, color=color, marker="D", s=60, edgecolor=INK_MUTED, zorder=3)
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.set_yticks(range(len(summary)), summary.index)
    ax.set_xlabel("Test Qini coefficient (dots = 5 seeds, diamond = mean)")
    ax.set_title("Stability across random splits — Hillstrom")
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-criteo", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.project.seed)
    figs, tables = cfg.path("figures"), cfg.path("tables")
    setup_mlflow(cfg, experiment="uplift-robustness")

    # 1. Seeds
    seeds = seed_stability(cfg)
    seeds.to_csv(tables / "robustness_seeds.csv", index=False)
    plot_seed_stability(seeds, figs / "robustness_seeds.png")
    seed_summary = (seeds.groupby("model")
                    .agg(qini_mean=("qini_coef", "mean"), qini_std=("qini_coef", "std"),
                         qini_min=("qini_coef", "min"), uplift10_mean=("uplift@10%", "mean"))
                    .sort_values("qini_mean", ascending=False))
    ranks = seeds.assign(rank=seeds.groupby("seed")["qini_coef"].rank(ascending=False))
    seed_summary["mean_rank"] = ranks.groupby("model")["rank"].mean()
    seed_summary["times_ranked_1st"] = ranks[ranks["rank"] == 1].groupby("model").size()
    seed_summary = seed_summary.fillna({"times_ranked_1st": 0})
    seed_summary.to_csv(tables / "robustness_seed_summary.csv")

    # 2. Men's email
    raw = fetch_hillstrom_raw(cfg)
    mens = encode_hillstrom(select_arms(raw, "mens"))
    mens_res = evaluate_split(cfg, mens, cfg.project.seed, HILLSTROM_MODELS, "hillstrom_mens", ci_iter=300)
    mens_res.to_csv(tables / "robustness_mens.csv", index=False)

    # 3. Criteo
    criteo_res = pd.DataFrame()
    if not args.skip_criteo:
        log.info("Loading Criteo (official file via Hugging Face, %d-row random sample)...", cfg.data.criteo.sample_rows)
        criteo = load_criteo(cfg)
        ate = estimate_ate(criteo[TARGET], criteo[TREATMENT])
        log.info("Criteo: %d rows, treated share %.2f, visit rate %.4f, ATE %+.4f",
                 len(criteo), criteo[TREATMENT].mean(), criteo[TARGET].mean(), ate.ate)
        criteo_res = evaluate_split(cfg, criteo, cfg.project.seed, list(cfg.robustness.criteo_models),
                                    "criteo", ci_iter=200)
        criteo_res["treated_share"] = criteo[TREATMENT].mean()
        criteo_res["ate"] = ate.ate
        criteo_res.to_csv(tables / "robustness_criteo.csv", index=False)

    for name, frame in [("hillstrom_mens", mens_res), ("criteo", criteo_res)]:
        for _, r in frame.iterrows():
            log_run(f"{name}/{r['model']}", {"dataset": name, "model": r["model"], "seed": int(r["seed"])},
                    {k: float(v) for k, v in r.items() if isinstance(v, float)}, tags={"module": "10"})
    for model, r in seed_summary.iterrows():
        log_run(f"seeds/{model}", {"dataset": "hillstrom", "model": model, "seeds": str(cfg.robustness.seeds)},
                {k: float(v) for k, v in r.items()}, tags={"module": "10"})

    pd.set_option("display.width", 160)
    print("\n=== Seed stability (Hillstrom, women's email)\n", seed_summary.round(4).to_string())
    cols = ["model", "qini_coef", "qini_ci_low", "qini_ci_high", "uplift@10%", "uplift@30%"]
    print("\n=== Men's email arm\n", mens_res.sort_values("qini_coef", ascending=False)[cols].round(4)
          .to_string(index=False))
    if len(criteo_res):
        print("\n=== Criteo (500k subsample)\n", criteo_res.sort_values("qini_coef", ascending=False)[cols]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
