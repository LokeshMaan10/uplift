"""Module 12 — "Who Will Respond?" Streamlit dashboard.

Run locally from the project root:
    streamlit run app/streamlit_app.py

Everything it needs lives in app/artifacts/ (written by scripts/run_pipeline.py and
scripts/run_explain.py), so the hosted app needs no data download and no API keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # makes `upliftlab` importable on Streamlit Cloud too

from upliftlab.business import (  # noqa: E402
    optimal_cutoff,
    profit_table,
    random_profit_table,
    uplift_at_percents,
    uplift_by_bins,
)
from upliftlab.segments import SegmentThresholds, assign_segments  # noqa: E402
from upliftlab.simulate import SEGMENTS  # noqa: E402

ART = ROOT / "app" / "artifacts"
PERCENTS = list(range(5, 101, 5))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9a9893"
SEGMENT_COLORS = {"Persuadable": "#1baf7a", "Sure Thing": "#2a78d6", "Lost Cause": NEUTRAL, "Sleeping Dog": "#eb6834"}
SEGMENT_ACTION = {
    "Persuadable": "✅ Send — the offer is likely to change their decision.",
    "Sure Thing": "💤 Skip — likely to respond anyway; the offer cost is wasted.",
    "Lost Cause": "⏭️ Skip — unlikely to respond either way.",
    "Sleeping Dog": "🚫 Do not send — the offer is predicted to reduce response.",
}

st.set_page_config(page_title="Who Will Respond? — Uplift Modeling", page_icon="🎯", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_tables() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((ART / "metadata.json").read_text())
    return (meta, pd.read_csv(ART / "model_comparison.csv"), pd.read_csv(ART / "curves.csv"),
            pd.read_csv(ART / "test_scores.csv"))


@st.cache_resource
def load_models() -> tuple[dict, object | None]:
    models = joblib.load(ART / "deploy_models.joblib")
    surrogate_path = ART / "surrogate.joblib"
    return models, joblib.load(surrogate_path) if surrogate_path.exists() else None


meta, comparison, curves, scores = load_tables()
models, surrogate = load_models()
FEATURES: list[str] = meta["features"]
DEPLOY: str = meta["deploy_model"]
BEST: str = meta["best_model"]
uplift_models = comparison.query("family != 'Baseline'")["model"].tolist()
all_models = comparison["model"].tolist()
y_test, t_test = scores["target"].to_numpy(), scores["treatment"].to_numpy()
ATE = float(meta["ate"])


def score_of(model: str) -> np.ndarray:
    return scores[f"score__{model}"].to_numpy()


def fmt_money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def md(text: str) -> str:
    """Escape '$' so Streamlit markdown shows money instead of rendering LaTeX."""
    return text.replace("$", r"\$")


def base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 420) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)), height=height,
                      xaxis_title=xtitle, yaxis_title=ytitle, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


# ---------------------------------------------------------------- sidebar
b = meta["business_defaults"]
with st.sidebar:
    st.title("🎯 Who Will Respond?")
    st.caption("Uplift modeling for targeted offers")
    st.subheader("Campaign economics")
    cost = st.slider("Cost per offer ($)", 0.0, 5.0, float(b["cost_per_offer"]), 0.05)
    value = st.slider("Profit per extra response ($)", 1.0, 100.0, float(b["value_per_response"]), 1.0)
    population = st.number_input("Customers in campaign", 1_000, 10_000_000, int(b["population_size"]), 10_000)
    st.caption(f"Break-even uplift = cost ÷ value = **{cost / value:.1%}**. "
               "Only customers whose uplift beats this are worth an offer.")
    st.divider()
    st.caption(f"Data: Hillstrom e-mail experiment ({meta['data']['treatment_mode']} e-mail vs none, "
               f"outcome = {meta['data']['target']}). Test set: {meta['n_test']:,} customers.")

tab_cmp, tab_roi, tab_seg, tab_one, tab_about = st.tabs(
    ["📊 Model comparison", "💰 ROI simulator", "🧩 Segments", "🙋 Should we send the offer?", "ℹ️ About"]
)

# ---------------------------------------------------------------- 1. comparison
with tab_cmp:
    best_row = comparison.iloc[0]
    resp_row = comparison.set_index("model").loc["Response model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", BEST)
    c2.metric("Qini coefficient (test)", f"{best_row['test_qini_coef']:.3f}",
              f"{best_row['test_qini_coef'] - resp_row['test_qini_coef']:+.3f} vs response model")
    c3.metric("Uplift in top 10%", f"{best_row['test_uplift@10%']:+.1%}", f"{best_row['test_uplift@10%'] - ATE:+.1%} vs average")
    c4.metric("Average effect (ATE)", f"{ATE:+.1%}")

    left, right = st.columns([3, 2])
    with left:
        kind = st.radio("Curve", ["qini", "uplift"], horizontal=True,
                        help="Both show extra responders gained as you target more customers, best-scored first.")
        default = list(dict.fromkeys([BEST, DEPLOY, "Response model"]))
        chosen = st.multiselect("Models (max 8)", all_models, default=default, max_selections=8)
        fig = go.Figure()
        sub = curves[curves["kind"] == kind]
        end = sub[sub["model"] == all_models[0]]["incremental_per_customer"].iloc[-1]
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, end * 100], name="Random targeting",
                                 line=dict(color=NEUTRAL, dash="dash", width=2)))
        for color, m in zip(SERIES, chosen):
            c = sub[sub["model"] == m]
            fig.add_trace(go.Scatter(x=c["frac_targeted"] * 100, y=c["incremental_per_customer"] * 100, name=m,
                                     line=dict(color=color, width=2.5),
                                     hovertemplate="%{y:.2f}% of customers"))
        st.plotly_chart(base_layout(fig, f"{kind.title()} curve (test set)", "% of customers targeted",
                                    "Incremental responders (% of customers)"), width="stretch")
        st.caption("Area between a curve and the dashed diagonal = value over random targeting. "
                   "A curve that rises steeply early finds Persuadables first.")
    with right:
        show = comparison[["model", "family", "test_qini_coef", "test_qini_ci_low", "test_qini_ci_high",
                           "cv_qini_coef_mean", "test_uplift@10%", "test_uplift@30%"]].copy()
        show.columns = ["Model", "Family", "Qini (test)", "CI low", "CI high", "Qini (5-fold CV)",
                        "Uplift@10%", "Uplift@30%"]
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[2:6]} |
                                       {"Uplift@10%": "{:+.1%}", "Uplift@30%": "{:+.1%}"})
                     .background_gradient(subset=["Qini (test)"], cmap="Blues"),
                     hide_index=True, width="stretch", height=400)
        dm = st.selectbox("Decile view for", all_models, index=all_models.index(BEST))
        dec = uplift_by_bins(y_test, score_of(dm), t_test, 10)
        fig = go.Figure(go.Bar(x=dec["bin"], y=dec["uplift"] * 100,
                               marker_color=[SERIES[0] if u >= 0 else SERIES[1] for u in dec["uplift"]],
                               error_y=dict(type="data", array=(dec["ci_high"] - dec["uplift"]) * 100,
                                            color="#52514e"),
                               hovertemplate="Decile %{x}: %{y:+.1f} pp<extra></extra>"))
        fig.add_hline(y=ATE * 100, line_dash="dash", line_color=NEUTRAL, annotation_text="average")
        st.plotly_chart(base_layout(fig, "Observed uplift by decile", "Decile (1 = highest score)",
                                    "Uplift (pp)", 300), width="stretch")

# ---------------------------------------------------------------- 2. ROI
with tab_roi:
    st.markdown("Profit if we send the offer to the **top X%** of customers ranked by each model. "
                "Uplift in each slice is measured on the held-out test set, then scaled to your campaign.")
    a, bcol = st.columns([2, 1])
    with a:
        roi_models = st.multiselect("Strategies", all_models,
                                    default=list(dict.fromkeys([BEST, DEPLOY, "Response model"])), max_selections=6)
    with bcol:
        pct = st.slider("% of customers targeted", 5, 100, 30, 5)

    tables = {m: profit_table(uplift_at_percents(y_test, score_of(m), t_test, PERCENTS), cost, value, population)
              for m in roi_models}
    rand = random_profit_table(ATE, PERCENTS, cost, value, population)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rand["percent"], y=rand["profit"], name="Random targeting",
                             line=dict(color=NEUTRAL, dash="dash", width=2)))
    for color, (m, tbl) in zip(SERIES, tables.items()):
        fig.add_trace(go.Scatter(x=tbl["percent"], y=tbl["profit"], name=m, line=dict(color=color, width=2.5),
                                 hovertemplate="$%{y:,.0f}"))
        opt = optimal_cutoff(tbl)
        if opt["percent"] > 0:
            fig.add_trace(go.Scatter(x=[opt["percent"]], y=[opt["profit"]], mode="markers", showlegend=False,
                                     marker=dict(size=12, color=color, line=dict(color="white", width=2)),
                                     hovertemplate=f"{m} optimum: %{{x}}% → $%{{y:,.0f}}<extra></extra>"))
    fig.add_vline(x=pct, line_color="#52514e", line_width=1)
    fig.add_hline(y=0, line_color="#52514e", line_width=1)
    st.plotly_chart(base_layout(fig, "Campaign profit by % targeted (dots = optimal cut-off)",
                                "% of customers targeted", "Profit ($)"), width="stretch")

    rows = []
    for m, tbl in tables.items():
        at = tbl[tbl["percent"] == pct].iloc[0]
        opt = optimal_cutoff(tbl)
        rows.append({"Strategy": m, f"Profit @ {pct}%": at["profit"], f"ROI @ {pct}%": at["roi"],
                     f"Extra responses @ {pct}%": at["incremental_responses"],
                     "Optimal %": opt["percent"], "Optimal profit": opt["profit"]})
    r_at = rand[rand["percent"] == pct].iloc[0]
    r_opt = optimal_cutoff(rand)  # linear profit -> optimum is 100% (or 0% if every offer loses)
    rows.append({"Strategy": "Random targeting", f"Profit @ {pct}%": r_at["profit"], f"ROI @ {pct}%": r_at["roi"],
                 f"Extra responses @ {pct}%": r_at["incremental_responses"], "Optimal %": r_opt["percent"],
                 "Optimal profit": r_opt["profit"]})
    everyone = rand.iloc[-1]
    rows.append({"Strategy": "Mail everyone", f"Profit @ {pct}%": everyone["profit"], f"ROI @ {pct}%": everyone["roi"],
                 f"Extra responses @ {pct}%": everyone["incremental_responses"], "Optimal %": 100,
                 "Optimal profit": everyone["profit"]})
    roi_df = pd.DataFrame(rows)
    st.dataframe(roi_df.style.format({f"Profit @ {pct}%": "${:,.0f}", f"ROI @ {pct}%": "{:.0%}",
                                      f"Extra responses @ {pct}%": "{:,.0f}", "Optimal %": "{:.0f}%",
                                      "Optimal profit": "${:,.0f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    if roi_models:
        top = roi_models[0]
        opt = optimal_cutoff(tables[top])
        gain = opt["profit"] - everyone["profit"]
        if opt["percent"] == 0:
            st.warning(md(f"At ${cost:.2f} per offer and ${value:.0f} per response, every targeting option loses "
                          "money — the best decision is **not to run this campaign**."))
        else:
            st.success(md(f"**{top}**: target the top **{opt['percent']:.0f}%** → profit "
                          f"**{fmt_money(opt['profit'])}**, {fmt_money(abs(gain))} "
                          f"{'more' if gain >= 0 else 'less'} than mailing everyone "
                          f"({fmt_money(everyone['profit'])}), with {100 - opt['percent']:.0f}% fewer offers sent."))

    with st.expander("What-if: how does the optimal plan change with offer cost?"):
        wm = st.selectbox("Model", all_models, index=all_models.index(BEST), key="whatif_model")
        upl = uplift_at_percents(y_test, score_of(wm), t_test, PERCENTS)
        wrows = []
        for c in [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5]:
            o = optimal_cutoff(profit_table(upl, c, value, population))
            wrows.append({"Cost per offer": c, "Optimal %": o["percent"], "Optimal profit": o["profit"],
                          "Mail-everyone profit": population * (ATE * value - c)})
        st.dataframe(pd.DataFrame(wrows).style.format({"Cost per offer": "${:.2f}", "Optimal %": "{:.0f}%",
                                                       "Optimal profit": "${:,.0f}",
                                                       "Mail-everyone profit": "${:,.0f}"}),
                     hide_index=True, width="stretch")
        st.caption("As offers get expensive, the model narrows targeting to the most persuadable customers; "
                   "mailing everyone turns loss-making much sooner.")

# ---------------------------------------------------------------- 3. segments
with tab_seg:
    th = SegmentThresholds(**meta["segment_thresholds"])
    uplift_dep = score_of(DEPLOY)
    seg = assign_segments(uplift_dep, scores["baseline_p0"].to_numpy(), th)
    segdf = scores.assign(segment=seg, uplift=uplift_dep)
    segdf["channel"] = segdf[[c for c in FEATURES if c.startswith("channel_")]].idxmax(axis=1).str.replace("channel_", "")
    segdf["zip_code"] = segdf[[c for c in FEATURES if c.startswith("zip_code_")]].idxmax(axis=1).str.replace("zip_code_", "")

    st.markdown(f"Customers labelled by the **{DEPLOY}**: predicted uplift ≥ {th.persuadable_uplift:.1%} → "
                f"Persuadable; < {th.sleeping_dog_uplift:.0%} → Sleeping Dog; otherwise baseline response ≥ "
                f"{th.sure_thing_baseline:.1%} → Sure Thing, else Lost Cause.")
    f1, f2 = st.columns(2)
    ch = f1.multiselect("Channel", sorted(segdf["channel"].unique()), default=sorted(segdf["channel"].unique()))
    zp = f2.multiselect("Area", sorted(segdf["zip_code"].unique()), default=sorted(segdf["zip_code"].unique()))
    view = segdf[segdf["channel"].isin(ch) & segdf["zip_code"].isin(zp)]

    if view.empty:
        st.info("No customers match these filters.")
    else:
        rows = []
        for s in SEGMENTS:
            p = view[view["segment"] == s]
            if len(p) and p["treatment"].nunique() == 2:
                r1, r0 = p.loc[p["treatment"] == 1, "target"].mean(), p.loc[p["treatment"] == 0, "target"].mean()
                n1, n0 = (p["treatment"] == 1).sum(), (p["treatment"] == 0).sum()
                se = np.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
                rows.append({"segment": s, "n": len(p), "share": len(p) / len(view), "predicted": p["uplift"].mean(),
                             "observed": r1 - r0, "ci": 1.96 * se})
            else:
                rows.append({"segment": s, "n": len(p), "share": len(p) / len(view), "predicted": np.nan,
                             "observed": np.nan, "ci": np.nan})
        st_df = pd.DataFrame(rows)

        l, r = st.columns(2)
        with l:
            fig = go.Figure(go.Bar(y=st_df["segment"], x=st_df["share"] * 100, orientation="h",
                                   marker_color=[SEGMENT_COLORS[s] for s in st_df["segment"]],
                                   text=[f"{v:.0%}" for v in st_df["share"]], textposition="outside",
                                   hovertemplate="%{y}: %{x:.1f}% (%{customdata:,} customers)<extra></extra>",
                                   customdata=st_df["n"]))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(fig, "Segment sizes", "% of customers", "", 320), width="stretch")
        with r:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=st_df["segment"], y=st_df["predicted"] * 100, name="Predicted",
                                 marker_color=NEUTRAL))
            fig.add_trace(go.Bar(x=st_df["segment"], y=st_df["observed"] * 100, name="Observed (treated − control)",
                                 marker_color=[SEGMENT_COLORS[s] for s in st_df["segment"]],
                                 error_y=dict(type="data", array=st_df["ci"] * 100, color="#52514e")))
            fig.update_layout(barmode="group")
            st.plotly_chart(base_layout(fig, "Does each segment behave as labelled?", "", "Uplift (pp)", 320),
                            width="stretch")
        st.caption("Observed uplift is measured from the randomized test set, so it independently checks the "
                   "model's labels. Small segments have wide error bars.")

        prof_cols = ["recency", "history", "mens", "womens", "newbie"]
        profile = view.groupby("segment")[prof_cols].mean().reindex(SEGMENTS)
        profile.loc["All (filtered)"] = view[prof_cols].mean()
        profile.columns = ["Months since last purchase", "Spend last year ($)", "Bought men's",
                           "Bought women's", "New customer"]
        st.subheader("Who is in each segment?")
        st.dataframe(profile.style.format({"Months since last purchase": "{:.1f}", "Spend last year ($)": "${:,.0f}",
                                           "Bought men's": "{:.0%}", "Bought women's": "{:.0%}",
                                           "New customer": "{:.0%}"}, na_rep="—"),
                     width="stretch")

        sample = view.sample(min(3000, len(view)), random_state=0)
        fig = go.Figure()
        for s in SEGMENTS:
            p = sample[sample["segment"] == s]
            fig.add_trace(go.Scatter(x=p["baseline_p0"] * 100, y=p["uplift"] * 100, mode="markers", name=s,
                                     marker=dict(color=SEGMENT_COLORS[s], size=6, opacity=0.6,
                                                 line=dict(color="white", width=0.5)),
                                     hovertemplate="baseline %{x:.1f}% · uplift %{y:+.1f} pp<extra>" + s + "</extra>"))
        fig.update_layout(hovermode="closest")
        st.plotly_chart(base_layout(fig, "Baseline response vs predicted uplift (sample of 3,000)",
                                    "P(respond | no offer) %", "Predicted uplift (pp)", 420),
                        width="stretch")

# ---------------------------------------------------------------- 4. single customer
with tab_one:
    st.markdown("Describe a customer and get a send / don't-send decision with the reasons behind it.")
    rng = meta["numeric_ranges"]
    c1, c2, c3 = st.columns(3)
    recency = c1.slider("Months since last purchase", int(rng["recency"][0]), int(rng["recency"][1]),
                        int(rng["recency"][2]))
    history = c1.number_input("Spend in past year ($)", 0.0, float(rng["history"][1]), float(rng["history"][2]), 10.0)
    bought = c2.radio("Bought last year", ["Women's only", "Men's only", "Both"], index=0)
    newbie = c2.toggle("New customer (joined in last 12 months)", value=True)
    zip_code = c3.selectbox("Area", ["Surburban", "Urban", "Rural"])
    channel = c3.selectbox("Purchase channel", ["Web", "Phone", "Multichannel"])

    row = {f: 0.0 for f in FEATURES}
    row.update({"recency": recency, "history": history, "newbie": int(newbie),
                "womens": int(bought in ("Women's only", "Both")), "mens": int(bought in ("Men's only", "Both")),
                f"zip_code_{zip_code}": 1, f"channel_{channel}": 1})
    x = pd.DataFrame([row])[FEATURES]

    up_model, base_model = models["uplift_model"], models["baseline_model"]
    uplift = float(up_model.predict_uplift(x)[0])
    if hasattr(up_model, "predict_outcomes"):
        p0, p1 = (float(v[0]) for v in up_model.predict_outcomes(x))
    else:
        p0 = float(base_model.predict_outcomes(x)[0][0])
        p1 = p0 + uplift
    seg_label = assign_segments(np.array([uplift]), np.array([float(base_model.predict_outcomes(x)[0][0])]), th)[0]
    exp_profit = uplift * value - cost

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P(visit) without e-mail", f"{p0:.1%}")
    k2.metric("P(visit) with e-mail", f"{p1:.1%}", f"{uplift:+.1%} uplift")
    k3.metric("Segment", seg_label)
    k4.metric("Expected profit per offer", f"{'-' if exp_profit < 0 else ''}${abs(exp_profit):.2f}")
    # Decision = economics (does expected extra profit beat the offer cost?).
    # The segment is descriptive context from fixed thresholds, so the two can differ.
    send = exp_profit > 0 and seg_label != "Sleeping Dog"
    math = f"Uplift {uplift:+.1%} × ${value:.0f} = ${uplift * value:.2f} expected gain vs ${cost:.2f} cost."
    if seg_label == "Sleeping Dog":
        reason = SEGMENT_ACTION[seg_label]
    elif send and seg_label == "Persuadable":
        reason = SEGMENT_ACTION[seg_label]
    elif send:
        reason = (f"Not in the top-30% '{seg_label}' group, but at these economics the offer still pays "
                  "for itself.")
    else:
        reason = f"The expected gain does not cover the offer cost. ({SEGMENT_ACTION[seg_label]})"
    (st.success if send else st.error)(md(
        f"**{'SEND the offer' if send else 'DO NOT send'}** — {reason}  {math}"
    ))

    if surrogate is not None:
        import shap

        sv = np.asarray(shap.TreeExplainer(surrogate).shap_values(x))[0]
        contrib = pd.DataFrame({"feature": FEATURES, "shap": sv})
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(6)
        labels = {"recency": f"Months since purchase = {recency}", "history": f"Spend = ${history:,.0f}",
                  "womens": f"Bought women's = {'yes' if row['womens'] else 'no'}",
                  "mens": f"Bought men's = {'yes' if row['mens'] else 'no'}",
                  "newbie": f"New customer = {'yes' if newbie else 'no'}"}
        contrib["label"] = [labels.get(f, f.replace("_", " = ", 1) + (" (yes)" if row[f] else " (no)"))
                            for f in contrib["feature"]]
        fig = go.Figure(go.Bar(y=contrib["label"][::-1], x=contrib["shap"][::-1] * 100, orientation="h",
                               marker_color=["#1baf7a" if v >= 0 else "#eb6834" for v in contrib["shap"][::-1]],
                               hovertemplate="%{x:+.2f} pp<extra></extra>"))
        st.plotly_chart(base_layout(fig, "Why? Features pushing this customer's uplift up (green) or down (orange)",
                                    "Contribution to predicted uplift (pp)", "", 340), width="stretch")
        r2 = meta.get("surrogate_r2_test")
        st.caption("SHAP values from a surrogate model that mimics the uplift model"
                   + (f" (fidelity R² = {r2:.2f})." if r2 is not None else "."))

# ---------------------------------------------------------------- 5. about
with tab_about:
    st.markdown(f"""
### The business problem
Sending an offer to everyone wastes money. A normal model predicts **who will respond**; an **uplift model**
predicts **who will respond *because* of the offer**:

> uplift(x) = P(respond | offer, x) − P(respond | no offer, x)

| Segment | Without offer | With offer | Action |
|---|---|---|---|
| Persuadable | ✗ | ✓ | **Target** |
| Sure Thing | ✓ | ✓ | Skip (wasted cost) |
| Lost Cause | ✗ | ✗ | Skip |
| Sleeping Dog | ✓ | ✗ | **Never target** |

### How it was built
- **Data:** Hillstrom e-mail experiment (randomized; balance verified, propensity AUC ≈ 0.50).
- **Models:** S/T/X/R meta-learners with LightGBM, class transformation, uplift random forest (causalml),
  causal forest (econml), vs random and response-model baselines.
- **Evaluation:** Qini & uplift curves, Qini coefficient, AUUC, uplift@k, deciles, 5-fold CV and
  bootstrap 95% CIs on a held-out test set.
- **This app** uses the **{DEPLOY}** for live predictions (fast, lightweight); the best test-set model was the
  **{BEST}**.

### Limitations
- Hillstrom's uplift signal is weak (ATE ≈ +4.5 pp, few features) so differences between models are
  modest and confidence intervals overlap.
- "Visit" is a proxy outcome; profit numbers use assumed economics (set them in the sidebar).
- Segments are model-based estimates, never directly observed for an individual.
""")
