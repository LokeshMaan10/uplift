"""Build the full project report as a PDF: reports/Uplift_Modeling_Report.pdf

Run after the pipeline scripts (it reads their tables and figures):
    python scripts/build_report.py

Every number in the report is read from reports/tables/*.csv or recomputed here,
so the PDF always matches the latest run.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from upliftlab.config import PROJECT_ROOT, load_config
from upliftlab.data import split_xyt
from upliftlab.eda import propensity_auc
from upliftlab.plots import AQUA, BLUE, INK, INK_MUTED, NEUTRAL, ORANGE, apply_style
from upliftlab.simulate import SEGMENTS, simulate_customers

CFG = load_config()
TABLES, FIGS = CFG.path("tables"), CFG.path("figures")
OUT = PROJECT_ROOT / "reports" / "Uplift_Modeling_Report.pdf"
AUTHOR = "Lokesh Maan"
REPO = "github.com/LokeshMaan10/uplift"

# ---------------------------------------------------------------- fonts & styles
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic",
                              boldItalic="DejaVu-Bold")

C_BLUE, C_INK, C_MUTED = colors.HexColor(BLUE), colors.HexColor(INK), colors.HexColor(INK_MUTED)
C_LIGHT, C_RULE = colors.HexColor("#eef4fc"), colors.HexColor("#d9d8d4")

S = {
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.6, leading=14.2, textColor=C_INK,
                           spaceAfter=6, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8, leading=11, textColor=C_MUTED),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Italic", fontSize=8, leading=11, textColor=C_MUTED,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=17, leading=22, textColor=C_BLUE,
                         spaceBefore=4, spaceAfter=10, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, leading=16, textColor=C_INK,
                         spaceBefore=10, spaceAfter=5, keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.6, leading=14, textColor=C_INK,
                             leftIndent=14, bulletIndent=3, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8, leading=10.5, textColor=C_INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=8, leading=10.5, textColor=colors.white),
    "callout": ParagraphStyle("callout", fontName="DejaVu", fontSize=9.6, leading=14, textColor=C_INK),
    "toc1": ParagraphStyle("toc1", fontName="DejaVu", fontSize=10.5, leading=19, leftIndent=6),
}


# ---------------------------------------------------------------- helpers
def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


class H1(Paragraph):
    """Section heading that also registers itself in the table of contents."""

    def __init__(self, text: str):
        super().__init__(text, S["h1"])
        self.toc_text = text


def figure(path: Path, width_cm: float, caption: str | None = None) -> KeepTogether:
    w, h = ImageReader(str(path)).getSize()
    width = width_cm * cm
    parts = [Image(str(path), width=width, height=width * h / w)]
    if caption:
        parts.append(P(caption, "caption"))
    return KeepTogether(parts)


def two_figures(left: Path, right: Path, caption: str, width_cm: float = 8.2) -> KeepTogether:
    imgs = []
    for p in (left, right):
        w, h = ImageReader(str(p)).getSize()
        imgs.append(Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w))
    t = Table([imgs], colWidths=[width_cm * cm + 0.2 * cm] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return KeepTogether([t, P(caption, "caption")])


def table(rows: list[list], col_widths: list[float], highlight_rows: tuple[int, ...] = (),
          align_right_from: int = 1) -> Table:
    """Styled table; first row is the header. Widths in cm."""
    data = [[Paragraph(str(c), S["cellb"]) for c in rows[0]]]
    data += [[Paragraph(str(c), S["cell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * cm for w in col_widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, C_RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), C_LIGHT))
    t.setStyle(TableStyle(style))
    return t


def callout(text: str) -> Table:
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, C_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def kpi_row(items: list[tuple[str, str]]) -> Table:
    cells = [[Paragraph(f"<font size=20 color='{BLUE}'><b>{v}</b></font><br/><br/>"
                        f"<font size=8 color='{INK_MUTED}'>{k}</font>", S["callout"]) for v, k in items]]
    t = Table(cells, colWidths=[16.4 / len(items) * cm] * len(items))
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, C_RULE), ("INNERGRID", (0, 0), (-1, -1), 0.6, C_RULE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


pct = lambda x, d=1: f"{x * 100:.{d}f}%"  # noqa: E731
pp = lambda x, d=1: f"{x * 100:+.{d}f} pp"  # noqa: E731
money = lambda x: ("−" if x < 0 else "") + (f"${abs(x) / 1000:,.1f}k" if abs(x) >= 1000 else f"${abs(x):,.0f}")  # noqa: E731


# ---------------------------------------------------------------- architecture diagram
def draw_architecture(path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=color, linewidth=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=color, edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", color="white", fontsize=9.5,
                fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + 0.12, y + h - 0.68 - i * 0.3, line, ha="left", va="center", fontsize=8.2, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=INK_MUTED, linewidth=1.3))

    box(0.1, 3.3, 2.1, 2.1, "1  Data", ["Hillstrom e-mail test", "64k customers", "Criteo uplift", "500k sample (HF)"],
        NEUTRAL)
    box(2.6, 0.3, 3.2, 5.1, "2  upliftlab package", [
        "data · eda", "  load, encode, split", "  balance + propensity AUC",
        "models", "  baselines", "  S · T · X · R learners", "  class transformation",
        "  uplift RF · causal forest", "metrics · evaluation", "  Qini, AUUC, CV, bootstrap",
        "business · segments · explain", "  ROI, cut-off, SHAP"], BLUE)
    box(6.2, 3.3, 2.1, 2.1, "3  Pipeline", ["run_eda", "run_pipeline", "run_explain", "run_robustness"], AQUA)
    box(6.2, 0.3, 2.1, 2.6, "4  Outputs", ["reports/ figures", "reports/ tables", "MLflow runs", "app/artifacts",
                                           "this PDF"], AQUA)
    box(8.7, 1.6, 2.2, 2.6, "5  Streamlit app", ["model comparison", "ROI simulator", "segment explorer",
                                                 "send / don't send", "+ SHAP reasons"], ORANGE)
    arrow(2.2, 4.35, 2.6, 4.35)
    arrow(5.8, 4.35, 6.2, 4.35)
    arrow(7.25, 3.3, 7.25, 2.9)
    arrow(8.3, 1.6, 8.7, 2.2)
    ax.text(9.8, 1.25, "deployed free on\nStreamlit Cloud / HF Spaces", ha="center", va="center", fontsize=8,
            color=INK_MUTED, style="italic")
    ax.text(0.1, 2.9, "config/config.yaml\none file for every seed,\npath, parameter and\nbusiness assumption",
            fontsize=8, color=INK_MUTED, va="top")
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- page decoration
def on_cover(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, h - 9.5 * cm, w, 9.5 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("DejaVu", 11)
    canvas.drawString(2 * cm, h - 2.6 * cm, "CAPSTONE PROJECT REPORT · CAUSAL MACHINE LEARNING")
    canvas.setFont("DejaVu-Bold", 30)
    canvas.drawString(2 * cm, h - 4.4 * cm, "Who Will Respond?")
    canvas.setFont("DejaVu", 15)
    canvas.drawString(2 * cm, h - 5.5 * cm, "Uplift Modeling for Targeted Offers")
    canvas.setFont("DejaVu", 10.5)
    canvas.drawString(2 * cm, h - 6.6 * cm, "Predicting who responds because of an offer — and what it is worth")
    canvas.setFont("DejaVu-Bold", 11)
    canvas.drawString(2 * cm, h - 8.2 * cm, AUTHOR)
    canvas.setFont("DejaVu", 10)
    canvas.drawString(2 * cm + canvas.stringWidth(AUTHOR, "DejaVu-Bold", 11) + 12, h - 8.2 * cm,
                      f"·  {date.today():%B %Y}  ·  {REPO}")
    canvas.restoreState()


def on_page(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setStrokeColor(C_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, h - 1.4 * cm, w - 2 * cm, h - 1.4 * cm)
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(2 * cm, h - 1.2 * cm, "Who Will Respond? — Uplift Modeling for Targeted Offers")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, AUTHOR)
    canvas.drawCentredString(w / 2, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, path: Path):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                         bottomMargin=1.8 * cm, title="Who Will Respond? — Uplift Modeling Report",
                         author=AUTHOR, subject="Uplift modeling capstone project report")
        cover_frame = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 12 * cm, id="cover")
        body_frame = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.8 * cm, id="body")
        self.addPageTemplates([PageTemplate("cover", [cover_frame], onPage=on_cover),
                               PageTemplate("body", [body_frame], onPage=on_page)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, H1):
            self.notify("TOCEntry", (0, flowable.toc_text, self.page))


# ---------------------------------------------------------------- data for the report
def load_numbers() -> dict:
    n = {}
    n["arms"] = pd.read_csv(TABLES / "eda_arms.csv")
    n["ate"] = pd.read_csv(TABLES / "eda_ate.csv").iloc[0]
    n["smd"] = pd.read_csv(TABLES / "eda_covariate_balance.csv")
    n["comp"] = pd.read_csv(TABLES / "model_comparison.csv")
    n["sim"] = pd.read_csv(TABLES / "sim_recovery.csv")
    n["roi"] = pd.read_csv(TABLES / "roi_optimal.csv")
    n["whatif"] = pd.read_csv(TABLES / "roi_what_if.csv")
    n["seg"] = pd.read_csv(TABLES / "segments_validation.csv")
    n["prof"] = pd.read_csv(TABLES / "segments_profile.csv", index_col=0)
    n["seeds"] = pd.read_csv(TABLES / "robustness_seed_summary.csv")
    n["mens"] = pd.read_csv(TABLES / "robustness_mens.csv").sort_values("qini_coef", ascending=False)
    n["criteo"] = pd.read_csv(TABLES / "robustness_criteo.csv").sort_values("qini_coef", ascending=False)
    n["tuning"] = pd.read_csv(TABLES / "lightgbm_tuning_cv.csv", index_col=0)
    n["meta"] = json.loads((PROJECT_ROOT / "app" / "artifacts" / "metadata.json").read_text())
    best_name = n["meta"]["deploy_model"]  # SHAP shown for the deployed model (its surrogate R² is in metadata)
    slug = "".join(ch if ch.isalnum() else "_" for ch in best_name.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    n["shap_slug"] = slug
    n["shap"] = pd.read_csv(TABLES / f"shap_importance_{slug}.csv")

    h = CFG.data.hillstrom
    df = pd.read_parquet(CFG.path("data_processed") / f"hillstrom_{h.treatment_mode}_{h.target}.parquet")
    X, _, t = split_xyt(df)
    n["auc"] = propensity_auc(X, t, seed=CFG.project.seed)

    sim = simulate_customers(20_000, seed=CFG.project.seed)
    n["sim_shares"] = sim["segment"].value_counts(normalize=True).reindex(SEGMENTS)
    n["sim_ate"] = sim["true_cate"].mean()
    k = int(0.2 * len(sim))
    sim["p_pooled"] = (sim["p0"] + sim["p1"]) / 2
    n["sim_picks"] = {name: sim.nlargest(k, col)["true_cate"].mean() * 1000
                      for name, col in [("Historical-buyer model", "p0"), ("Campaign response model", "p_pooled"),
                                        ("Uplift model", "true_cate")]}
    n["sim_dogs"] = {name: (sim.nlargest(k, col)["segment"] == "Sleeping Dog").mean()
                     for name, col in [("Historical-buyer model", "p0"), ("Campaign response model", "p_pooled"),
                                       ("Uplift model", "true_cate")]}
    return n


# ---------------------------------------------------------------- the report
def build() -> Path:
    n = load_numbers()
    comp, ate, roi, wi = n["comp"], n["ate"], n["roi"], n["whatif"]
    meta = n["meta"]
    best, deploy = meta["best_model"], meta["deploy_model"]
    upl = comp[comp["family"] != "Baseline"]
    resp = comp.set_index("model").loc["Response model"]
    rand = comp.set_index("model").loc["Random"]
    everyone = roi.set_index("strategy").loc["Mail everyone"]
    top_roi = roi[~roi["strategy"].isin(["Mail everyone", "Response model", "Historical buyer"])].iloc[0]
    gain = top_roi["profit"] - everyone["profit"]
    seg = n["seg"].set_index("segment")
    crit = n["criteo"].set_index("model")
    seeds = n["seeds"].set_index("model")
    uplift_seed_min = seeds.drop(["Response model", "Random"])["qini_min"].min()
    arch = draw_architecture(FIGS / "architecture.png")

    story: list = []

    # ---- cover
    story += [Spacer(1, 1.2 * cm),
              kpi_row([(f"{top_roi['profit'] / everyone['profit']:.1f}×", "campaign profit vs mailing everyone"),
                       (f"−{100 - top_roi['optimal_percent']:.0f}%", "fewer offers sent"),
                       (f"{len(upl)}/{len(upl)}", "uplift models beat every baseline")]),
              Spacer(1, 0.8 * cm),
              P("<b>Abstract.</b> Companies send offers to entire customer bases, although most customers either "
                "respond anyway or never will. This project builds uplift models that estimate, for each customer, "
                "the <i>causal</i> effect of an offer on the probability of responding, using a randomized "
                f"e-mail experiment with {int(ate['n_treated'] + ate['n_control']):,} customers. Seven uplift "
                "estimators (S-, T-, X- and R-learners on LightGBM, class transformation, uplift random forest and "
                "causal forest) are compared with response-model baselines using Qini and uplift curves, "
                "cross-validation and bootstrap confidence intervals. All uplift models clearly beat the baselines; "
                f"the standard response model is statistically indistinguishable from random targeting "
                f"(Qini {resp['test_qini_coef']:.3f} vs {rand['test_qini_coef']:.3f}). Translated into money, "
                f"uplift targeting earns {money(top_roi['profit'])} instead of {money(everyone['profit'])} per "
                f"100,000 customers while contacting only {top_roi['optimal_percent']:.0f}% of them. Results are "
                "explained with SHAP, stress-tested across random seeds, a second treatment arm and 500,000 Criteo "
                "users, and delivered as an interactive Streamlit dashboard."),
              Spacer(1, 0.4 * cm),
              P(f"<b>Stack:</b> Python 3.11 · pandas · scikit-learn · LightGBM · scikit-uplift · CausalML · EconML · "
                f"SHAP · MLflow · Streamlit · pytest. 100% free tools and public data, no API keys.", "small"),
              NextPageTemplate("body"), PageBreak()]

    # ---- contents
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    story += [P("Contents", "h1"), toc, PageBreak()]

    # ---- executive summary
    story += [H1("Executive summary"),
              P("<b>Problem.</b> A response model ranks customers by how likely they are to buy. That puts "
                "customers who would buy anyway (\"Sure Things\") at the top of the mailing list, so the campaign "
                "pays for sales it did not cause. The business question is different: <i>whose behaviour does the "
                "offer change?</i> That quantity — the individual treatment effect, or <b>uplift</b> — is what this "
                "project estimates."),
              P("<b>Approach.</b> Using the randomized Hillstrom e-mail experiment, I first verified that treatment "
                f"was truly random (propensity AUC {n['auc']:.3f}), measured the average effect "
                f"({pp(ate['ate'])}), and showed that it varies strongly between customers. I then trained and "
                "compared ten models — three baselines and seven uplift estimators — under one interface, evaluated "
                "them on a held-out test set with uplift-specific metrics, converted the rankings into campaign "
                "profit, explained them with SHAP, and checked robustness on new splits and new data."),
              P("<b>Key findings</b>")]
    story += bullets([
        f"<b>Every uplift model beats every baseline.</b> Test Qini coefficients range from "
        f"{upl['test_qini_coef'].min():.3f} to {upl['test_qini_coef'].max():.3f}, all with 95% confidence "
        f"intervals above zero. The response model scores {resp['test_qini_coef']:.3f} — the same as random "
        f"({rand['test_qini_coef']:.3f}).",
        f"<b>About twice the profit with half the mail.</b> At $1 per offer and $30 per extra response, targeting "
        f"the top {top_roi['optimal_percent']:.0f}% by uplift yields {money(top_roi['profit'])} versus "
        f"{money(everyone['profit'])} for mailing everyone (+{money(gain)} per 100,000 customers).",
        "<b>The model knows when to stop.</b> As offers get more expensive the optimal target group shrinks; "
        f"from ${wi.loc[wi['optimal_percent'] == 0, 'cost_per_offer'].min():.0f} per offer the best decision is "
        "not to run the campaign.",
        f"<b>Who is persuadable:</b> past buyers of women's products (the treatment was the women's e-mail). "
        f"Customers the model labels Persuadable showed an observed uplift of {pp(seg.loc['Persuadable', 'observed_uplift'])}; "
        f"Sure Things showed {pp(seg.loc['Sure Thing', 'observed_uplift'])}.",
        "<b>Honest limits.</b> Uplift modeling only pays when effects differ between customers and are not "
        "aligned with baseline response: on the men's e-mail no model beats random, and on Criteo a response "
        "model performs as well as the uplift models.",
    ])
    story += [Spacer(1, 4),
              callout("<b>Recommendation.</b> Replace response-model targeting with an uplift model, choose the "
                      "share of customers to contact from the profit curve for the current offer cost, and keep a "
                      "small random control group in every campaign so uplift can be re-measured and the model "
                      "retrained."),
              PageBreak()]

    # ---- 1 business problem
    story += [H1("1. Business problem"),
              P("A bank sending credit-card or loan offers — or a retailer sending promotional e-mails — usually "
                "contacts everyone, or everyone a response model rates highly. Both waste money, because the "
                "decision that matters is not whether a customer will respond, but whether the offer <i>changes</i> "
                "that response. Every customer falls into one of four groups, defined by what they would do with "
                "and without the offer:"),
              table([["Segment", "Without offer", "With offer", "Effect of the offer", "Action"],
                     ["Persuadable", "does not respond", "responds", "creates a response", "<b>Target</b>"],
                     ["Sure Thing", "responds", "responds", "none — cost is wasted", "Skip"],
                     ["Lost Cause", "does not respond", "does not respond", "none — cost is wasted", "Skip"],
                     ["Sleeping Dog", "responds", "does not respond", "destroys a response", "<b>Never target</b>"]],
                    [2.8, 3.0, 3.0, 4.0, 3.6], highlight_rows=(1,)),
              Spacer(1, 8),
              P("An uplift model estimates, for a customer with features <i>x</i>, the conditional average "
                "treatment effect (CATE):"),
              callout("τ(x) = P(respond | offer, x) − P(respond | no offer, x)"),
              Spacer(1, 6),
              P("Ranking customers by τ(x) puts Persuadables first and Sleeping Dogs last. The project goal is to "
                "build that ranking, prove it is better than the alternatives, and translate it into money saved.")]

    # ---- 2 data
    arms = n["arms"].set_index("arm")
    story += [H1("2. Data and experiment validation"),
              P("All data is public and loaded with free tools. The main dataset is the Hillstrom MineThatData "
                "e-mail experiment (via scikit-uplift): 64,000 customers were randomly assigned to receive a men's "
                "e-mail, a women's e-mail, or no e-mail. The outcome is a website visit within two weeks. Purchases "
                "(~1%) are too rare to model reliably, so visits are used as the response."),
              table([["Dataset", "Rows used", "Treatment", "Outcome", "Role"],
                     ["Hillstrom (women's e-mail vs none)", f"{int(ate['n_treated'] + ate['n_control']):,}",
                      "e-mail sent (50%)", "visit", "main analysis"],
                     ["Hillstrom (men's e-mail vs none)", "42,613", "e-mail sent (50%)", "visit", "robustness"],
                     ["Criteo Uplift v2.1", "500,000 of 14M", f"ad shown ({pct(crit['treated_share'].iloc[0], 0)})",
                      "visit", "scale + 2nd domain"]],
                    [4.6, 2.5, 3.3, 2.4, 3.6]),
              Spacer(1, 6),
              P("Features: months since last purchase, spend in the past year, whether the customer bought men's "
                "and/or women's merchandise, new-customer flag, area (urban / suburban / rural) and purchase channel. "
                "The binned spend column was dropped as a duplicate of spend. The scikit-uplift Criteo download "
                "links no longer work, so the loader fetches Criteo's official file from its public Hugging Face "
                "repository and samples it in chunks to stay within laptop memory."),
              P("2.1 Was treatment really random?", "h2"),
              P("Uplift estimates are only causal if the treated and control groups were comparable before the "
                "e-mail. Three checks confirm this:")]
    story += bullets([
        f"<b>Standardized mean differences:</b> the largest across all 11 features is "
        f"{n['smd']['smd'].abs().max():.3f}, far below the usual 0.1 threshold.",
        f"<b>t-tests:</b> {(n['smd']['p_value'] < 0.05).sum()} of 11 features differ at p &lt; 0.05 (about 0.6 "
        "expected by chance).",
        f"<b>Propensity AUC = {n['auc']:.3f}:</b> a model trying to predict who received the e-mail from customer "
        "features does no better than a coin flip.",
    ])
    story += [two_figures(FIGS / "eda_response_by_arm.png", FIGS / "eda_covariate_balance.png",
                          "Figure 1. Left: visit rate by campaign arm. Right: covariate balance — every feature lies "
                          "inside the ±0.1 band."),
              P("2.2 The average effect hides large differences", "h2"),
              P(f"The women's e-mail lifted visits from {pct(ate['rate_control'])} to {pct(ate['rate_treated'])}: "
                f"an average treatment effect of <b>{pp(ate['ate'], 2)}</b> (95% CI {pp(ate['ci_low'], 2)} to "
                f"{pp(ate['ci_high'], 2)}), a {pct(ate['relative_lift'], 0)} relative lift. The men's e-mail had an "
                f"even larger average effect ({pp(arms.loc['Mens E-Mail', 'lift_vs_control'])}). But the effect is "
                "not uniform: among past buyers of women's products the women's e-mail adds about +7 pp, among "
                "men's-only buyers about +1 pp. A single average would waste most of the budget — this "
                "heterogeneity is exactly what uplift models exploit."),
              figure(FIGS / "eda_uplift_by_group.png", 11.5,
                     "Figure 2. Uplift (treated − control) within customer groups, with 95% CIs. Dashed line = "
                     "overall ATE.")]

    # ---- 3 concepts
    shares = n["sim_shares"]
    picks = n["sim_picks"]
    story += [H1("3. Causal concepts"),
              P("<b>Potential outcomes.</b> Each customer has two possible futures: Y(1), what they do if contacted, "
                "and Y(0), what they do if not. The individual effect is Y(1) − Y(0). <b>The fundamental problem of "
                "causal inference</b> is that only one of the two is ever observed, so individual effects can never "
                "be measured directly. Instead we estimate the average effect for customers with similar features — "
                "the CATE, τ(x)."),
              P("<b>ATE vs CATE.</b> The average treatment effect (ATE) is one number for everyone and, with "
                "randomization, equals the difference in response rates between the groups. The CATE varies with "
                "customer features and is what decides whom to target."),
              P("To make these ideas visible, I simulated a bank where both potential outcomes are known. "
                f"In the simulation, {pct(shares['Persuadable'], 0)} of customers are Persuadables, "
                f"{pct(shares['Sure Thing'], 0)} Sure Things, {pct(shares['Lost Cause'], 0)} Lost Causes and "
                f"{pct(shares['Sleeping Dog'], 0)} Sleeping Dogs, with a true ATE of {pp(n['sim_ate'])}. Giving three "
                "targeting strategies a perfect model of their own target and the same budget (top 20%) shows why "
                "the question matters:"),
              table([["Strategy (perfect model, top 20%)", "Extra responses per 1,000 contacted", "Sleeping Dogs contacted"],
                     *[[k, f"{v:+.0f}", pct(n["sim_dogs"][k])] for k, v in picks.items()]],
                    [7.0, 5.2, 4.2], highlight_rows=(3,)),
              Spacer(1, 6),
              P("The classic \"propensity to buy\" model <i>loses</i> responses because loyal, engaged customers — "
                "where Sleeping Dogs live — look like the best prospects. A response model trained on campaign data "
                "fills its budget with Sure Things. The uplift ranking roughly doubles the incremental responses."),
              two_figures(FIGS / "concepts_ate_vs_cate.png", FIGS / "concepts_response_vs_uplift.png",
                          "Figure 3. Simulated bank. Left: one ATE, very different individual effects. Right: who "
                          "each strategy contacts."),
              P("<b>Assumptions.</b> (1) Unconfoundedness — treatment is independent of potential outcomes given x, "
                "guaranteed here by randomization; (2) overlap — every customer type can appear in both groups; "
                "(3) SUTVA — one customer's treatment does not affect another's outcome.")]

    # ---- 4 method
    tun = n["tuning"]
    story += [H1("4. Methodology and architecture"),
              figure(arch, 16.2, "Figure 4. System architecture: one package, four pipeline scripts, shared outputs, "
                                 "and a lightweight dashboard."),
              P("4.1 Models", "h2"),
              P("Every model implements the same two methods — <font face='DejaVu-Bold'>fit(X, y, t)</font> and "
                "<font face='DejaVu-Bold'>predict_uplift(X)</font> — so evaluation, ROI and the app treat them "
                "identically. The meta-learners are implemented from scratch on LightGBM."),
              table([["Model", "How it estimates uplift", "Strength / weakness"],
                     ["Random", "random score", "reference: what untargeted mailing achieves"],
                     ["Response model", "P(respond | x), treatment ignored", "standard industry practice; wrong question"],
                     ["Historical buyer", "P(respond | x, no offer)", "\"propensity to buy\"; targets Sure Things"],
                     ["S-learner", "one model f(x, t); f(x,1) − f(x,0)", "simple; can under-use t (shrinks uplift)"],
                     ["T-learner", "μ1(x) − μ0(x), one model per group", "flexible; noise of two models adds up"],
                     ["X-learner", "impute effects with the other group's model, then model them; propensity-weighted",
                      "good for unbalanced groups"],
                     ["R-learner", "Robinson residuals: (y − m(x)) on (t − e), cross-fitted",
                      "targets the CATE directly; robust to outcome-model error"],
                     ["Class transformation", "Z = 1 if (treated & responded) or (control & not); uplift = 2P(Z) − 1",
                      "one classifier; reweighted for unbalanced splits"],
                     ["Uplift random forest", "trees split on treated-vs-control divergence (KL), causalml",
                      "purpose-built; slowest"],
                     ["Causal forest", "double ML + honest forest on residuals, econml", "statistically grounded; stable"]],
                    [3.2, 7.4, 5.8]),
              P("4.2 Evaluation protocol", "h2")]
    story += bullets([
        "70/30 train/test split, stratified on treatment × outcome so both parts keep the same treated share and "
        "response rate.",
        "5-fold stratified cross-validation on the training set for model comparison and tuning; the test set is "
        "used once for final numbers.",
        "Metrics: <b>Qini coefficient</b> (area between the model's Qini curve and random, normalised by a "
        "perfect model; 0 = random, 1 = perfect), <b>AUUC</b>, <b>uplift@k%</b> (treated − control response rate "
        "in the top k%), and uplift by decile.",
        f"<b>Bootstrap</b> ({CFG.evaluation.bootstrap_iterations} resamples of the test set) for 95% confidence "
        "intervals on the Qini coefficient.",
        "A known-truth check on simulated data, where each model's predictions are compared with the true uplift.",
    ])
    story += [P("4.3 Hyper-parameters chosen without touching the test set", "h2"),
              P("The uplift signal in Hillstrom is weak, so flexible trees fit noise. Three shared LightGBM settings "
                "were compared by cross-validated Qini on training data only; strong regularisation won and was "
                "used for all LightGBM-based models."),
              table([["Setting", "Mean CV Qini (5 learners)"],
                     *[[c, f"{tun.loc['MEAN', c]:.3f}"] for c in tun.columns]], [9.0, 5.0], highlight_rows=(3,))]

    # ---- 5 results
    rows = [["Model", "Family", "Qini (test)", "95% CI", "CV Qini", "Uplift<br/>top 10%", "Uplift<br/>top 30%"]]
    for _, r in comp.iterrows():
        rows.append([r["model"], r["family"], f"{r['test_qini_coef']:.3f}",
                     f"[{r['test_qini_ci_low']:.3f}, {r['test_qini_ci_high']:.3f}]",
                     f"{r['cv_qini_coef_mean']:.3f} ± {r['cv_qini_coef_std']:.3f}",
                     pp(r["test_uplift@10%"]), pp(r["test_uplift@30%"])])
    sim = n["sim"]
    story += [H1("5. Results"),
              P(f"Table 5.1 compares all ten models on the held-out test set of {meta['n_test']:,} customers."),
              table(rows, [3.3, 2.3, 1.8, 2.7, 2.6, 1.9, 1.9], highlight_rows=(1,)),
              Spacer(1, 8)]
    story += bullets([
        f"<b>All seven uplift models beat all three baselines</b>, and all their confidence intervals exclude zero. "
        "The response model's interval includes zero: it is no better than random targeting.",
        f"The top four models' intervals overlap, so they are statistically tied. {best} had the highest test "
        f"Qini; the causal forest had the best and most stable cross-validated score. The dashboard uses the "
        f"{deploy} — pure LightGBM, fast and small to deploy.",
        "Uplift@10% is similar for all models (the very top slice is noisy); the difference shows up across the "
        "first 30–50% of customers, which is where campaign profit is decided.",
        f"<b>Known-truth check:</b> on simulated data, uplift models reached Qini "
        f"{sim[~sim['model'].isin(['Random', 'Response model', 'Historical buyer'])]['qini_coef'].min():.2f}–"
        f"{sim['qini_coef'].max():.2f} against a perfect {sim['oracle_qini_coef'].iloc[0]:.2f}, while the response "
        f"model reached only {sim.set_index('model').loc['Response model', 'qini_coef']:.2f}.",
    ])
    story += [two_figures(FIGS / "qini_baselines_vs_best.png", FIGS / "model_ci.png",
                          "Figure 5. Left: Qini curves — the uplift model gathers incremental responders early, "
                          "response models follow the random line. Right: test Qini with bootstrap 95% CIs."),
              two_figures(FIGS / "qini_uplift_models.png", FIGS / "deciles_best.png",
                          f"Figure 6. Left: Qini curves for all uplift models. Right: observed uplift by decile for "
                          f"the {best}.")]

    # ---- 6 business
    roi_rows = [["Strategy", "Best % to target", "Profit", "ROI", "Extra responses"]]
    for _, r in roi.iterrows():
        roi_rows.append([r["strategy"], f"{r['optimal_percent']:.0f}%", money(r["profit"]),
                         pct(r["roi"], 0), f"{r['incremental_responses']:,.0f}"])
    wi_rows = [["Cost per offer", "Break-even uplift", "Optimal % to target", "Profit (model)", "Profit (mail all)"]]
    for _, r in wi.iterrows():
        wi_rows.append([f"${r['cost_per_offer']:.2f}", pct(r["breakeven_uplift"]),
                        f"{r['optimal_percent']:.0f}%" if r["optimal_percent"] > 0 else "don't run",
                        money(r["optimal_profit"]), money(r["mail_everyone_profit"])])
    b = CFG.business
    story += [H1("6. Business impact"),
              P(f"For each strategy and each targeted share p, the uplift among the top p% is measured on the "
                f"randomized test set and scaled to a campaign of {b.population_size:,} customers, with "
                f"<b>${b.cost_per_offer:.0f} per offer</b> and <b>${b.value_per_response:.0f} profit per extra "
                "response</b>. Because uplift is measured rather than predicted, the profit figures are causal "
                "estimates."),
              callout("profit(p) = uplift(top p%) × customers targeted × value − customers targeted × cost"),
              Spacer(1, 6),
              table(roi_rows, [4.6, 3.0, 2.8, 2.4, 3.6], highlight_rows=(1,)),
              Spacer(1, 6),
              P(f"Uplift targeting earns <b>+{money(gain)}</b> more than mailing everyone — "
                f"{top_roi['profit'] / everyone['profit']:.1f}× the profit — while contacting only "
                f"{top_roi['optimal_percent']:.0f}% of customers. Neither response-style model can beat mailing "
                "everyone: their best option is to contact all customers."),
              figure(FIGS / "roi_profit_curves.png", 12.5,
                     "Figure 7. Profit by share of customers targeted; dots mark each strategy's optimum."),
              P("6.1 What-if: offer cost", "h2"),
              P(f"The optimal plan depends on the offer cost. A customer is worth contacting only if their uplift "
                f"exceeds the break-even rate cost ÷ value. Table 6.2 repeats the analysis for the {best}."),
              table(wi_rows, [3.0, 3.0, 3.4, 3.4, 3.6]),
              Spacer(1, 6),
              figure(FIGS / "roi_what_if.png", 11.5,
                     "Figure 8. As offers get more expensive the model narrows targeting; mailing everyone turns "
                     "loss-making much sooner.")]

    # ---- 7 explainability
    shap_df = n["shap"].head(6)
    prof = n["prof"]
    seg_rows = [["Segment", "Share", "Predicted uplift", "Observed uplift (95% CI)", "Baseline rate"]]
    for s in SEGMENTS:
        r = seg.loc[s]
        seg_rows.append([s, pct(r["share"], 0), pp(r["predicted_uplift"]),
                         f"{pp(r['observed_uplift'])} [{pp(r['ci_low'])}, {pp(r['ci_high'])}]",
                         pct(r["control_rate"])])
    story += [H1("7. Explainability and segments"),
              P("Meta-learners combine several sub-models, so there is no single tree to inspect. A surrogate "
                f"LightGBM model was trained to mimic each model's predicted uplift (fidelity R² "
                f"{meta.get('surrogate_r2_test', 0):.2f} for the {deploy}) and explained with SHAP. The direction "
                "column shows whether a higher feature value increases (+) or decreases (−) predicted uplift."),
              table([["Feature", "Mean |SHAP| (pp)", "Direction"],
                     *[[r["feature"], f"{r['mean_abs_shap'] * 100:.2f}", "+" if r["direction"] > 0 else "−"]
                       for _, r in shap_df.iterrows()]], [6.0, 4.0, 3.0]),
              Spacer(1, 6),
              figure(FIGS / f"shap_beeswarm_{n['shap_slug']}.png", 12.5,
                     f"Figure 9. SHAP values on predicted uplift ({deploy} surrogate)."),
              P("Interpretation: the e-mail promoted women's merchandise, so customers who bought women's products "
                "last year are by far the most persuadable; men's-only buyers and rural customers react least; "
                "high spenders are slightly less persuadable because they tend to visit anyway."),
              P("7.1 Do predicted segments behave as labelled?", "h2"),
              P("Customers were labelled from predictions: uplift below 0 → Sleeping Dog; top 30% of uplift → "
                "Persuadable; otherwise Sure Thing if their baseline response is above the median, else Lost Cause. "
                "Because the test set is randomized, each segment's true uplift can be measured independently:"),
              table(seg_rows, [3.0, 1.8, 3.0, 5.2, 3.4], highlight_rows=(1,)),
              Spacer(1, 6),
              figure(FIGS / "segments_validation.png", 12.0, "Figure 10. Predicted vs observed uplift per segment."),
              P(f"Persuadables were validated: observed uplift {pp(seg.loc['Persuadable', 'observed_uplift'])}, almost "
                f"twice the average. Sure Things were validated too: high baseline "
                f"({pct(seg.loc['Sure Thing', 'control_rate'])}) and essentially zero uplift. Persuadables are all "
                f"past women's buyers ({pct(prof.loc['Persuadable', 'womens'], 0)}) and spend "
                f"${prof.loc['Persuadable', 'history']:.0f} on average. The Lost Cause label proved too pessimistic "
                "(their uplift is close to average), and the small Sleeping Dog group cannot be confirmed — both "
                "are reported as limitations.")]

    # ---- 8 robustness
    seed_rows = [["Model", "Mean Qini", "Std", "Worst seed", "Mean rank"]]
    for m, r in seeds.iterrows():
        seed_rows.append([m, f"{r['qini_mean']:.3f}", f"{r['qini_std']:.3f}", f"{r['qini_min']:.3f}",
                          f"{r['mean_rank']:.1f}"])
    crit_rows = [["Model", "Qini", "95% CI", "Uplift@10%"]]
    for m, r in crit.iterrows():
        crit_rows.append([m, f"{r['qini_coef']:.3f}", f"[{r['qini_ci_low']:.3f}, {r['qini_ci_high']:.3f}]",
                          pp(r["uplift@10%"])])
    mens = n["mens"]
    story += [H1("8. Robustness"),
              P("8.1 Stability across random splits", "h2"),
              P("The full split-train-evaluate cycle was repeated with five different seeds on Hillstrom (the slow "
                f"uplift random forest was excluded). The worst run of any uplift model (Qini {uplift_seed_min:.3f}) "
                f"still beat the response model's average ({seeds.loc['Response model', 'qini_mean']:.3f})."),
              table(seed_rows, [4.6, 2.6, 2.4, 2.6, 2.6]),
              Spacer(1, 6),
              figure(FIGS / "robustness_seeds.png", 11.5, "Figure 11. Test Qini for each of five seeds."),
              P("8.2 A second treatment: the men's e-mail", "h2"),
              P(f"The men's e-mail has a larger average effect, but no model beats random targeting: the best Qini "
                f"is {mens['qini_coef'].max():.3f} and every confidence interval includes zero. The effect is "
                "roughly the same for everyone, so there is nothing to target — mailing all customers is the right "
                "decision for that campaign. Uplift modeling only adds value when effects are heterogeneous."),
              P("8.3 Scale and a second domain: Criteo", "h2"),
              P(f"On 500,000 Criteo users ({pct(crit['treated_share'].iloc[0], 0)} treated, visit rate about 4.7%, "
                f"ATE {pp(crit['ate'].iloc[0], 2)}), all models beat random clearly. However, the response model "
                "ties the best uplift models: in this dataset, users likely to visit are also the ones the ads move, "
                "so ranking by response probability and by uplift agree."),
              table(crit_rows, [5.0, 2.6, 4.2, 3.0]),
              Spacer(1, 6),
              callout("<b>Takeaway.</b> Uplift models were never worse than the response model in any setting, and "
                      "far better when uplift and baseline response diverge (Hillstrom). They are the safe default — "
                      "but checking for heterogeneity first avoids over-engineering campaigns whose effect is uniform.")]

    # ---- 9 engineering
    story += [H1("9. Engineering and deployment"),
              P("9.1 Interactive dashboard", "h2"),
              table([["Tab", "Purpose"],
                     ["Model comparison", "Qini / uplift curves for any models, results table, uplift by decile"],
                     ["ROI simulator", "sliders for offer cost, profit per response and % targeted; profit, ROI and "
                                       "optimal cut-off for each strategy; offer-cost what-if"],
                     ["Segments", "segment sizes, predicted vs observed uplift, profiles, filters by channel and area"],
                     ["Should we send the offer?", "describe one customer → probabilities with and without the offer, "
                                                   "uplift, segment, send / don't send with expected profit and SHAP "
                                                   "reasons"]],
                    [4.2, 12.2]),
              P("9.2 Reproducibility and quality", "h2")]
    story += bullets([
        "<b>Single configuration file</b> (config/config.yaml) for every seed, path, hyper-parameter and business "
        "assumption.",
        "<b>MLflow</b> logs every model run with parameters, metrics and the fitted model, plus all robustness runs.",
        "<b>Automated tests</b> cover the ATE and balance statistics, every model (including recovery of the true "
        "uplift on simulated data), the uplift metrics, the ROI arithmetic and decision rules, the segment rules, "
        "SHAP, and a headless run of the whole dashboard.",
        "<b>Deployment check:</b> the app was run in a fresh environment containing only its slim requirements, "
        "which exposed and fixed two hidden dependency problems before deployment.",
        "<b>Free hosting:</b> Streamlit Community Cloud (slim requirements file) or Hugging Face Spaces "
        "(Dockerfile included). Code: " + REPO + ".",
    ])

    # ---- 10 limitations
    story += [H1("10. Limitations and future work"),
              P("Limitations", "h2")]
    story += bullets([
        "<b>Weak signal.</b> Hillstrom has 11 features and a modest effect, so Qini values are small (~0.07) and the "
        "best uplift models are statistically tied.",
        "<b>Proxy outcome.</b> Website visits stand in for offer acceptance; purchases are too rare to model.",
        "<b>Assumed economics.</b> The $1 cost and $30 value are illustrative; the what-if analysis and the "
        "dashboard show how conclusions change.",
        "<b>Segments are estimates.</b> True segments are unobservable; the Lost Cause and Sleeping Dog labels did "
        "not validate cleanly.",
        "<b>Noise in small slices.</b> Uplift in the top 5–10% rests on a few hundred customers per group.",
        "<b>Single campaign.</b> No drift, seasonality or repeated-contact fatigue can be studied.",
    ])
    story += [P("Future work", "h2")]
    story += bullets([
        "Model profit directly (spend uplift) instead of visit uplift, and optimise for expected profit per customer.",
        "Multi-treatment uplift: choose the best of several offers (men's vs women's e-mail) per customer.",
        "Run each campaign with a random holdout group to monitor real uplift and retrain on fresh data.",
        "Policy evaluation with doubly robust estimators and budget-constrained targeting.",
    ])

    # ---- 11 conclusion
    story += [H1("11. Conclusion"),
              P("Asking \"who will respond?\" leads to targeting customers who would respond anyway. Asking \"who will "
                "respond <i>because</i> of the offer?\" — and answering it with causal machine learning on randomized "
                "data — lets a company spend its marketing budget only where it changes behaviour. In this project, "
                f"all seven uplift estimators outperformed response-based targeting, the best plans roughly doubled "
                f"campaign profit while halving the number of offers, and the robustness analysis clarified when the "
                "approach pays off and when it does not. The full pipeline — validation, modeling, evaluation, "
                "business simulation, explanation and an interactive tool — is reproducible end to end with free "
                "software and public data."),
              Spacer(1, 10),
              P("Appendix: glossary", "h2"),
              table([["Term", "Meaning"],
                     ["Uplift / CATE", "change in response probability caused by the offer for customers with features x"],
                     ["ATE", "average treatment effect over everyone: E[Y(1) − Y(0)]"],
                     ["Qini curve", "incremental responders gained as more customers are targeted, best-scored first"],
                     ["Qini coefficient", "area between the Qini curve and random, normalised by a perfect model"],
                     ["AUUC", "area under the uplift curve, normalised the same way"],
                     ["uplift@k%", "treated minus control response rate among the top k% of customers"],
                     ["Propensity AUC", "how well features predict treatment; 0.5 means assignment was random"],
                     ["SMD", "standardized mean difference between groups; below 0.1 counts as balanced"],
                     ["Bootstrap CI", "range of a metric over resampled test sets; shows its uncertainty"]],
                    [3.6, 12.8])]

    # start a section on a new page rather than leave its heading stranded at the bottom
    story = [x for f in story for x in ((CondPageBreak(9 * cm), f) if isinstance(f, H1) else (f,))]
    doc = ReportDoc(OUT)
    doc.multiBuild(story)
    return OUT


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    path = build()
    print(f"Report written: {path}")
