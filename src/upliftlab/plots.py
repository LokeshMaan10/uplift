"""Matplotlib figures for the report. One consistent, quiet style for every chart."""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed palette: identity colours never change between charts
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#9a9893"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# Categorical series order (fixed, never cycled; max 8 lines per chart)
SERIES: list[str] = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# The 4 customer segments keep the same colour in every chart and in the app
SEGMENT_COLORS: dict[str, str] = {
    "Persuadable": AQUA,
    "Sure Thing": BLUE,
    "Lost Cause": NEUTRAL,
    "Sleeping Dog": ORANGE,
}

STYLE: dict = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,         # gridlines behind bars
        "text.parse_math": False,       # '$100' is money, not LaTeX
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
}


def apply_style() -> None:
    """Reset matplotlib to the project style.

    Needed because some libraries (causalml) switch the global style on import.
    Every plot_* function below calls this automatically.
    """
    plt.style.use("default")
    plt.rcParams.update(STYLE)


apply_style()


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_response_by_arm(arm_table: pd.DataFrame, control_label: str, path: Path) -> Path:
    """Bar of response rate per Hillstrom arm with 95% CI whiskers.

    arm_table columns: arm, rate, ci_low, ci_high (rates as fractions).
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [NEUTRAL if a == control_label else BLUE for a in arm_table["arm"]]
    err = [arm_table["rate"] - arm_table["ci_low"], arm_table["ci_high"] - arm_table["rate"]]
    ax.bar(arm_table["arm"], arm_table["rate"] * 100, color=colors, width=0.55,
           yerr=[e * 100 for e in err], capsize=4, ecolor=INK_MUTED)
    for x, r in enumerate(arm_table["rate"]):
        ax.text(x, r * 100 / 2, f"{r:.1%}", ha="center", color="white", fontweight="bold")
    ax.set_ylabel("Response rate (%)")
    ax.set_title("Response rate by campaign arm")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_covariate_balance(smd: pd.DataFrame, threshold: float, path: Path) -> Path:
    """'Love plot': SMD per feature; dots inside the ±threshold band = balanced."""
    data = smd.sort_values("smd", key=abs)
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(data) + 1.2))
    ax.axvspan(-threshold, threshold, color=AQUA, alpha=0.10, lw=0)
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.scatter(data["smd"], data["feature"], s=36, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.5)
    lim = max(threshold * 1.5, data["smd"].abs().max() * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardized mean difference (treated − control)")
    ax.set_title(f"Covariate balance — shaded band = |SMD| < {threshold}")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_uplift_by_groups(tables: dict[str, pd.DataFrame], ate: float, path: Path) -> Path:
    """Small multiples: ATE (pp) with 95% CI within each level of several columns.

    Dashed line = overall ATE. Bars far from it hint at who is persuadable.
    """
    n = len(tables)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 1.0 + sum(len(t) for t in tables.values()) * 0.42),
                             sharex=True,  # same scale in every panel so bars are comparable
                             gridspec_kw={"height_ratios": [len(t) for t in tables.values()]})
    for ax, (col, tbl) in zip(axes, tables.items()):
        labels = tbl[col].astype(str)
        err = [(tbl["ate"] - tbl["ci_low"]) * 100, (tbl["ci_high"] - tbl["ate"]) * 100]
        ax.barh(labels, tbl["ate"] * 100, color=BLUE, height=0.6,
                xerr=err, capsize=3, ecolor=INK_MUTED)
        ax.axvline(ate * 100, color=ORANGE, ls="--", lw=1.5, label=f"Overall ATE ({ate:+.1%})")
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[0].set_title("Uplift (treated − control, pp) by customer group")
    axes[-1].legend(loc="center right", frameon=False, fontsize=9)
    axes[-1].set_xlabel("Uplift in response rate (percentage points)")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- Module 7: evaluation
def plot_curves(curves: dict[str, pd.DataFrame], title: str, ylabel: str, path: Path) -> Path:
    """Qini/uplift curves for up to 8 models + the random-targeting diagonal.

    curves[name] has columns frac_targeted, incremental_per_customer.
    """
    if len(curves) > len(SERIES):
        raise ValueError("At most 8 series per chart — split into panels.")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    end = next(iter(curves.values()))["incremental_per_customer"].iloc[-1]
    ax.plot([0, 100], [0, end * 100], color=NEUTRAL, ls="--", lw=1.5, label="Random targeting")
    for color, (name, c) in zip(SERIES, curves.items()):
        ax.plot(c["frac_targeted"] * 100, c["incremental_per_customer"] * 100, color=color, lw=2, label=name)
    ax.set_xlabel("% of customers targeted (highest score first)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, path)


def plot_metric_ci(table: pd.DataFrame, metric: str, low: str, high: str, title: str, path: Path) -> Path:
    """Horizontal bars of a metric with bootstrap CI whiskers, best model on top."""
    data = table.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(data) + 1.2))
    colors = [NEUTRAL if f == "Baseline" else BLUE for f in data["family"]]
    err = [data[metric] - data[low], data[high] - data[metric]]
    ax.barh(data["model"], data[metric], color=colors, height=0.6, xerr=err, capsize=3, ecolor=INK_MUTED)
    ax.axvline(0, color=INK_MUTED, lw=1)
    for i, v in enumerate(data[metric]):
        ax.text(max(v, 0) + 0.005, i + 0.28, f"{v:.3f}", fontsize=8, color=INK_MUTED)
    ax.set_xlabel(f"{metric} (95% bootstrap CI) — grey = baseline")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_deciles(table: pd.DataFrame, model_name: str, ate: float, path: Path) -> Path:
    """Observed uplift per predicted-score decile; should fall from left to right."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [BLUE if u >= 0 else ORANGE for u in table["uplift"]]
    ax.bar(table["decile"], table["uplift"] * 100, color=colors, width=0.65,
           yerr=table["std_uplift"] * 196, capsize=3, ecolor=INK_MUTED)
    ax.axhline(ate * 100, color=INK_MUTED, ls="--", lw=1.2, label=f"Average effect {ate:+.1%}")
    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.set_xticks(table["decile"])
    ax.set_xlabel("Decile of predicted uplift (1 = most persuadable)")
    ax.set_ylabel("Observed uplift (pp)")
    ax.set_title(f"{model_name}: observed uplift by decile (test set)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


# ---------------------------------------------------------------- Module 8: business
def plot_profit_curves(tables: dict[str, pd.DataFrame], currency: str, path: Path) -> Path:
    """Profit vs % targeted for several strategies; dots mark each optimum."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = {"Random targeting": NEUTRAL}
    palette = iter(SERIES)
    for name, tbl in tables.items():
        color = colors.get(name) or next(palette)
        ls = "--" if name == "Random targeting" else "-"
        ax.plot(tbl["percent"], tbl["profit"] / 1000, color=color, lw=2, ls=ls, label=name)
        if name != "Random targeting":
            best = tbl.loc[tbl["profit"].idxmax()]
            ax.scatter(best["percent"], best["profit"] / 1000, s=60, color=color, zorder=4,
                       edgecolor="white", linewidth=2)
    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.set_xlabel("% of customers targeted")
    ax.set_ylabel(f"Campaign profit ({currency} thousands)")
    ax.set_title("Profit by targeting strategy — dots = optimal cut-off")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_what_if(df: pd.DataFrame, currency: str, path: Path) -> Path:
    """Best-model optimal profit vs mail-everyone profit as offer cost changes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["cost_per_offer"], df["optimal_profit"] / 1000, color=BLUE, lw=2, marker="o",
            label="Uplift model, optimal cut-off")
    ax.plot(df["cost_per_offer"], df["mail_everyone_profit"] / 1000, color=NEUTRAL, lw=2, ls="--",
            marker="o", label="Mail everyone")
    for _, r in df.iterrows():
        ax.annotate(f"{r['optimal_percent']:.0f}%", (r["cost_per_offer"], r["optimal_profit"] / 1000),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.set_xlabel(f"Cost per offer ({currency})")
    ax.set_ylabel(f"Profit ({currency} thousands)")
    ax.set_title("What-if: offer cost vs profit (labels = optimal % to target)")
    ax.legend(frameon=False)
    return _save(fig, path)


# ---------------------------------------------------------------- Module 9: explainability
def plot_importance(imp: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    """Mean |SHAP| bars; colour shows direction (aqua: higher value -> more uplift)."""
    data = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(data) + 1.3))
    colors = [AQUA if d >= 0 else ORANGE for d in data["direction"]]
    ax.barh(data["feature"], data["mean_abs_shap"] * 100, color=colors, height=0.6)
    ax.set_xlabel("Mean |SHAP| on predicted uplift (pp)")
    ax.set_title("What drives persuadability?  aqua: higher value → more uplift · orange: less")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_segment_validation(table: pd.DataFrame, path: Path) -> Path:
    """Predicted vs observed uplift per predicted segment (test set)."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(table))
    ax.bar(x - 0.18, table["predicted_uplift"] * 100, width=0.36, color="white", edgecolor=INK_MUTED,
           hatch="///", linewidth=1, label="Predicted")
    ax.bar(x + 0.18, table["observed_uplift"] * 100, width=0.36,
           color=[SEGMENT_COLORS[s] for s in table["segment"]], label="Observed (treated − control)",
           yerr=(table["ci_high"] - table["observed_uplift"]) * 100, capsize=3, ecolor=INK_MUTED)
    ax.axhline(0, color=INK_MUTED, lw=1)
    ax.set_xticks(x, [f"{s}\n({share:.0%})" for s, share in zip(table["segment"], table["share"])])
    ax.set_ylabel("Uplift (pp)")
    ax.set_title("Do predicted segments behave as labelled? (test set)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


# ---------------------------------------------------------------- style guard
def _styled(fn):
    """Wrap a plot function so it always starts from the project style."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        apply_style()
        return fn(*args, **kwargs)

    return wrapper


for _name, _fn in list(globals().items()):
    if _name.startswith("plot_") and callable(_fn):
        globals()[_name] = _styled(_fn)
