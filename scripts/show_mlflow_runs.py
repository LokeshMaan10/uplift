"""Print the MLflow runs logged by the pipeline (Module 11 check).

Run:  python scripts/show_mlflow_runs.py
UI:   mlflow ui --backend-store-uri mlruns   ->  http://127.0.0.1:5000
"""

import mlflow
import pandas as pd

from upliftlab.config import load_config


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.path("mlruns").as_uri())
    pd.set_option("display.width", 160)
    for exp in mlflow.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"\n=== Experiment '{exp.name}': {len(runs)} runs")
        cols = [c for c in ["tags.mlflow.runName", "metrics.test_qini_coef", "metrics.cv_qini_coef_mean",
                            "metrics.qini_coef", "metrics.qini_mean"] if c in runs.columns]
        if len(runs):
            print(runs[cols].round(4).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
