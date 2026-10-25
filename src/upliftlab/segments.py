"""Label customers as Persuadable / Sure Thing / Lost Cause / Sleeping Dog from predictions.

True segments need both potential outcomes (never observed), so we use the
model's estimates instead:
    predicted uplift < sleeping_dog_uplift           -> Sleeping Dog (offer predicted to hurt)
    predicted uplift >= persuadable threshold         -> Persuadable
    otherwise, baseline P(respond | no offer) high    -> Sure Thing
    otherwise                                         -> Lost Cause
Thresholds are learned on the TRAINING predictions and reused everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from upliftlab.simulate import LOST_CAUSE, PERSUADABLE, SLEEPING_DOG, SURE_THING


@dataclass(frozen=True)
class SegmentThresholds:
    persuadable_uplift: float
    sleeping_dog_uplift: float
    sure_thing_baseline: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_thresholds(
    uplift: np.ndarray,
    baseline: np.ndarray,
    persuadable_quantile: float = 0.7,
    sleeping_dog_uplift: float = 0.0,
    sure_thing_quantile: float = 0.5,
) -> SegmentThresholds:
    """Derive cut-offs from training-set predictions."""
    return SegmentThresholds(
        persuadable_uplift=float(np.quantile(uplift, persuadable_quantile)),
        sleeping_dog_uplift=float(sleeping_dog_uplift),
        sure_thing_baseline=float(np.quantile(baseline, sure_thing_quantile)),
    )


def assign_segments(uplift: np.ndarray, baseline: np.ndarray, th: SegmentThresholds) -> np.ndarray:
    """Vectorised segment label for every customer."""
    uplift, baseline = np.asarray(uplift), np.asarray(baseline)
    return np.select(
        [
            uplift < th.sleeping_dog_uplift,
            uplift >= th.persuadable_uplift,
            baseline >= th.sure_thing_baseline,
        ],
        [SLEEPING_DOG, PERSUADABLE, SURE_THING],
        default=LOST_CAUSE,
    )
