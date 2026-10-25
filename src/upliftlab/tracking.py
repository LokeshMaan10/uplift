"""Module 11 — MLflow experiment tracking (local file store, no server needed).

View runs:  mlflow ui --backend-store-uri mlruns   ->  http://127.0.0.1:5000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from upliftlab.config import Config


def setup_mlflow(cfg: Config, experiment: str | None = None) -> None:
    """Point MLflow at ./mlruns and select (or create) the experiment."""
    mlflow.set_tracking_uri(cfg.path("mlruns").as_uri())
    mlflow.set_experiment(experiment or cfg.mlflow.experiment_name)


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> flat 'a.b' keys (MLflow params must be flat)."""
    flat: dict[str, Any] = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one model run; returns the MLflow run id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(params))
        # MLflow metric names cannot contain '%' or '@'
        mlflow.log_metrics({k.replace("@", "_at_").replace("%", "pct"): v for k, v in metrics.items()})
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        if tags:
            mlflow.set_tags(tags)
        return run.info.run_id
