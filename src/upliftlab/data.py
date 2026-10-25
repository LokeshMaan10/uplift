"""Load uplift datasets via scikit-uplift and turn them into model-ready frames.

Every loader returns one DataFrame with:
    - feature columns (numeric, one-hot encoded)
    - TREATMENT column: 1 = received offer, 0 = control
    - TARGET column:    1 = responded,     0 = did not
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom
from tqdm import tqdm

from upliftlab.config import Config

TREATMENT = "treatment"
TARGET = "target"

# Hillstrom arm labels exactly as sklift returns them
MENS, WOMENS, CONTROL = "Mens E-Mail", "Womens E-Mail", "No E-Mail"
TREATED_ARMS: dict[str, list[str]] = {
    "any_email": [MENS, WOMENS],
    "womens": [WOMENS],
    "mens": [MENS],
}
HILLSTROM_CATEGORICALS: list[str] = ["zip_code", "channel"]


# ---------------------------------------------------------------- Hillstrom
def fetch_hillstrom_raw(cfg: Config) -> pd.DataFrame:
    """Download (or read cached) Hillstrom; keep all 3 arms and raw columns.

    Columns: original features + 'arm' (text label) + TARGET (0/1).
    """
    bunch = fetch_hillstrom(
        target_col=cfg.data.hillstrom.target,
        data_home=str(cfg.path("data_raw")),
    )
    df = bunch.data.copy()
    df["arm"] = bunch.treatment.values
    df[TARGET] = bunch.target.values.astype(int)
    return df


def select_arms(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Keep the treated arm(s) for `mode` plus control; add binary TREATMENT."""
    if mode not in TREATED_ARMS:
        raise ValueError(f"treatment_mode must be one of {list(TREATED_ARMS)}, got '{mode}'")
    treated = TREATED_ARMS[mode]
    df = raw[raw["arm"].isin(treated + [CONTROL])].copy()
    df[TREATMENT] = df["arm"].isin(treated).astype(int)
    return df.reset_index(drop=True)


def encode_hillstrom(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and drop columns models must not see.

    - history_segment is just a binned copy of `history` -> dropped (redundant)
    - arm is the raw treatment label -> dropped (TREATMENT replaces it)
    """
    out = df.drop(columns=["history_segment", "arm"])
    out = pd.get_dummies(out, columns=HILLSTROM_CATEGORICALS, dtype=int)
    # Clean names for LightGBM / SHAP (no spaces or symbols)
    out.columns = [c.replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def load_hillstrom(cfg: Config) -> pd.DataFrame:
    """Model-ready Hillstrom: encoded features + TREATMENT + TARGET."""
    raw = fetch_hillstrom_raw(cfg)
    return encode_hillstrom(select_arms(raw, cfg.data.hillstrom.treatment_mode))


# ---------------------------------------------------------------- Criteo
CRITEO_FEATURES: list[str] = [f"f{i}" for i in range(12)]
CRITEO_TOTAL_ROWS = 13_979_592  # rows in criteo-research-uplift-v2.1


def download_file(url: str, dest: Path, chunk_mb: int = 4) -> Path:
    """Stream a public file to disk once (cached); shows a progress bar."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def load_criteo(cfg: Config) -> pd.DataFrame:
    """Criteo uplift (randomized ad exposure, ~14M users), sampled to `sample_rows`.

    The 311 MB file is read in 1M-row chunks and each chunk keeps a random
    fraction of rows, so memory stays small. Features f0..f11 are anonymised
    numerics. Note the split is ~85% treated / 15% control. Used in Module 10.
    """
    c = cfg.data.criteo
    path = download_file(c.url, cfg.path("data_raw") / "criteo" / Path(c.url).name)
    frac = min(1.0, c.sample_rows / CRITEO_TOTAL_ROWS * 1.02)  # small margin, trimmed below
    rng = np.random.default_rng(cfg.project.seed)
    parts = []
    usecols = CRITEO_FEATURES + ["treatment", c.target]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000, dtype="float32"):
        parts.append(chunk[rng.random(len(chunk)) < frac])
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(n=min(c.sample_rows, len(df)), random_state=cfg.project.seed)
    df = df.rename(columns={"treatment": TREATMENT, c.target: TARGET})
    df[TREATMENT] = df[TREATMENT].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df[CRITEO_FEATURES + [TREATMENT, TARGET]].reset_index(drop=True)


# ---------------------------------------------------------------- helpers
def _strata(df: pd.DataFrame) -> pd.Series:
    """Joint treatment x outcome label (4 groups) for stratified splitting."""
    return df[TREATMENT].astype(str) + "_" + df[TARGET].astype(str)


def split_xyt(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a model-ready frame into features X, outcome y, treatment t."""
    X = df.drop(columns=[TREATMENT, TARGET])
    return X, df[TARGET], df[TREATMENT]


def train_test_split_uplift(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split stratified on treatment x outcome.

    Keeps the treated/control ratio AND response rate equal in both parts,
    which uplift metrics need (they compare treated vs control inside each slice).
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=_strata(df), random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
