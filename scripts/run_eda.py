"""Module 2 — Data & EDA on Hillstrom.

Run from the project root (venv active):
    python scripts/run_eda.py

Outputs:
    reports/tables/eda_*.csv           numbers for the report
    reports/figures/eda_*.png          figures for the report / README
    data/processed/hillstrom_*.parquet model-ready data used by later modules
"""

from __future__ import annotations

import pandas as pd

from upliftlab.config import load_config
from upliftlab.data import (
    CONTROL,
    TARGET,
    TREATMENT,
    encode_hillstrom,
    fetch_hillstrom_raw,
    select_arms,
    split_xyt,
)
from upliftlab.eda import (
    SMD_THRESHOLD,
    class_balance,
    estimate_ate,
    propensity_auc,
    standardized_mean_differences,
    uplift_by_group,
)
from upliftlab.plots import plot_covariate_balance, plot_response_by_arm, plot_uplift_by_groups
from upliftlab.utils import get_logger, set_seed

# Raw (pre-encoding) columns used to look for heterogeneous effects
HETEROGENEITY_COLS: list[str] = ["history_segment", "channel", "zip_code", "newbie", "womens", "mens"]

pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)


def arm_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Response rate of every arm, and its lift vs the no-email control."""
    control = raw[raw["arm"] == CONTROL]
    rows = []
    for arm, part in raw.groupby("arm"):
        if arm == CONTROL:
            p = part[TARGET].mean()
            se = (p * (1 - p) / len(part)) ** 0.5
            rows.append({"arm": arm, "n": len(part), "rate": p, "ci_low": p - 1.96 * se,
                         "ci_high": p + 1.96 * se, "lift_vs_control": 0.0, "p_value": float("nan")})
            continue
        pair = pd.concat([part, control])
        res = estimate_ate(pair[TARGET], (pair["arm"] == arm).astype(int))
        se = (res.rate_treated * (1 - res.rate_treated) / res.n_treated) ** 0.5
        rows.append({"arm": arm, "n": len(part), "rate": res.rate_treated,
                     "ci_low": res.rate_treated - 1.96 * se, "ci_high": res.rate_treated + 1.96 * se,
                     "lift_vs_control": res.ate, "p_value": res.p_value})
    order = {CONTROL: 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
    return pd.DataFrame(rows).sort_values("arm", key=lambda s: s.map(order), ignore_index=True)


def main() -> None:
    cfg = load_config()
    set_seed(cfg.project.seed)
    log = get_logger("eda")
    hcfg = cfg.data.hillstrom
    figs, tables = cfg.path("figures"), cfg.path("tables")

    # 1. Load ----------------------------------------------------------------
    log.info("Loading Hillstrom (first run downloads ~4 MB)...")
    raw = fetch_hillstrom_raw(cfg)
    print(f"\n=== Raw data: {raw.shape[0]:,} rows x {raw.shape[1]} cols | outcome = '{hcfg.target}'")
    print(raw.head())
    print("\nMissing values per column:", int(raw.isna().sum().sum()))
    print("\nDtypes:\n", raw.dtypes.to_string())

    # 2. All three arms -------------------------------------------------------
    arms = arm_summary(raw)
    print("\n=== Response rate by arm (lift = vs No E-Mail)")
    print(arms.to_string(index=False))
    arms.to_csv(tables / "eda_arms.csv", index=False)
    plot_response_by_arm(arms, CONTROL, figs / "eda_response_by_arm.png")

    # 3. Binary treatment for modeling ---------------------------------------
    df = select_arms(raw, hcfg.treatment_mode)
    model_df = encode_hillstrom(df)
    X, y, t = split_xyt(model_df)
    print(f"\n=== Modeling frame (treatment_mode='{hcfg.treatment_mode}'): "
          f"{len(model_df):,} rows, {X.shape[1]} features")
    print("Features:", list(X.columns))

    # 4. Randomization checks -------------------------------------------------
    smd = standardized_mean_differences(X, t)
    auc = propensity_auc(X, t, seed=cfg.project.seed, folds=cfg.split.cv_folds)
    print(f"\n=== Covariate balance (|SMD| < {SMD_THRESHOLD} = balanced)")
    print(smd.to_string(index=False))
    print(f"\nMax |SMD|: {smd['smd'].abs().max():.4f} | "
          f"features with p<0.05: {(smd['p_value'] < 0.05).sum()} of {len(smd)} "
          f"(expect ~{0.05 * len(smd):.1f} by chance)")
    print(f"Propensity AUC (can features predict treatment?): {auc:.3f}  -> ~0.50 means randomized")
    smd.to_csv(tables / "eda_covariate_balance.csv", index=False)
    plot_covariate_balance(smd, SMD_THRESHOLD, figs / "eda_covariate_balance.png")

    # 5. Class balance + ATE --------------------------------------------------
    cb = class_balance(y, t)
    ate = estimate_ate(y, t)
    print("\n=== Class balance")
    print(cb)
    print("\n=== Average Treatment Effect (ATE)")
    print(f"Response: treated {ate.rate_treated:.2%} vs control {ate.rate_control:.2%}")
    print(f"ATE = {ate.ate:+.2%} points  (95% CI {ate.ci_low:+.2%} to {ate.ci_high:+.2%}), "
          f"relative lift {ate.relative_lift:+.1%}, "
          f"p {'< 1e-15' if ate.p_value < 1e-15 else f'= {ate.p_value:.2e}'}")
    cb.to_csv(tables / "eda_class_balance.csv")
    pd.DataFrame([ate.to_dict()]).to_csv(tables / "eda_ate.csv", index=False)

    # 6. Heterogeneity preview -----------------------------------------------
    by_group = {c: uplift_by_group(df, c, TARGET, TREATMENT) for c in HETEROGENEITY_COLS}
    print("\n=== Uplift by group (preview of heterogeneous effects)")
    for col, tbl in by_group.items():
        print(f"\n-- {col}")
        print(tbl[[col, "n", "rate_control", "rate_treated", "ate", "ci_low", "ci_high"]]
              .to_string(index=False))
    pd.concat([tbl.rename(columns={c: "level"}).assign(column=c) for c, tbl in by_group.items()]
              ).to_csv(tables / "eda_uplift_by_group.csv", index=False)
    plot_uplift_by_groups(by_group, ate.ate, figs / "eda_uplift_by_group.png")

    # 7. Save model-ready data ------------------------------------------------
    out = cfg.path("data_processed") / f"hillstrom_{hcfg.treatment_mode}_{hcfg.target}.parquet"
    model_df.to_parquet(out, index=False)
    log.info("Saved %s | figures -> %s | tables -> %s", out.name, figs, tables)


if __name__ == "__main__":
    main()
