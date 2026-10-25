"""Synthetic bank customers where we KNOW both potential outcomes.

Real data only shows one outcome per customer (offer OR no offer). A simulation
shows both, so we can see the four segments directly, compute the true CATE,
and later check whether our models recover it.

Story: a bank sends a credit-card offer.
    - Engaged customers often apply anyway        -> high baseline (Sure Things)
    - Young, mid-income, low-engagement customers -> offer helps a lot (Persuadables)
    - Long-tenure, highly engaged customers find it pushy -> offer hurts (Sleeping Dogs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSUADABLE, SURE_THING, LOST_CAUSE, SLEEPING_DOG = (
    "Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog",
)
SEGMENTS: list[str] = [PERSUADABLE, SURE_THING, LOST_CAUSE, SLEEPING_DOG]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def simulate_customers(n: int = 20_000, seed: int = 42, treat_prob: float = 0.5) -> pd.DataFrame:
    """Simulate customers with features, both potential outcomes and a random treatment.

    Columns:
        age, income_k, engagement, tenure_years      features
        p0, p1        P(apply | no offer), P(apply | offer)
        true_cate     p1 - p0 (the individual uplift we try to estimate)
        y0, y1        potential outcomes (in real life you only ever see one)
        segment       Persuadable / Sure Thing / Lost Cause / Sleeping Dog
        treatment     randomly assigned (0/1)
        target        observed outcome = y1 if treated else y0
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 76, n)
    income_k = np.clip(rng.lognormal(np.log(55), 0.45, n), 10, 300)
    engagement = rng.beta(2, 3, n)  # 0..1, app logins / product usage score
    tenure_years = np.clip(rng.exponential(5, n), 0, 30)

    # Baseline propensity to apply without any offer
    p0 = _sigmoid(-2.6 + 3.2 * engagement + 0.04 * tenure_years)

    # Effect of the offer on the logit scale (heterogeneous by design)
    young = (age < 40).astype(float)
    mid_income = ((income_k > 35) & (income_k < 90)).astype(float)
    shift = (
        1.6 * young * mid_income * (1 - engagement)     # persuadable pocket
        + 0.3                                           # small general effect
        - 2.5 * (engagement > 0.5) * (tenure_years > 6)  # pushy for loyal, active customers
    )
    p1 = _sigmoid(_logit(p0) + shift)

    # One shared random draw per customer: "how close they are to applying".
    # This makes y0/y1 consistent (no one is both a Sure Thing and a Lost Cause by noise).
    u = rng.random(n)
    y0 = (u < p0).astype(int)
    y1 = (u < p1).astype(int)

    segment = np.select(
        [(y0 == 0) & (y1 == 1), (y0 == 1) & (y1 == 1), (y0 == 0) & (y1 == 0)],
        [PERSUADABLE, SURE_THING, LOST_CAUSE],
        default=SLEEPING_DOG,
    )
    treatment = (rng.random(n) < treat_prob).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "income_k": income_k.round(1),
            "engagement": engagement.round(3),
            "tenure_years": tenure_years.round(1),
            "p0": p0,
            "p1": p1,
            "true_cate": p1 - p0,
            "y0": y0,
            "y1": y1,
            "segment": segment,
            "treatment": treatment,
            "target": np.where(treatment == 1, y1, y0),
        }
    )


FEATURES: list[str] = ["age", "income_k", "engagement", "tenure_years"]
