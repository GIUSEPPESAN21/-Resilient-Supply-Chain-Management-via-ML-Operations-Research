"""Optional MLflow experiment tracking for rolling-origin backtesting runs: logs
hyperparameters, per-fold metrics (mean +/- std), and a calibration-plot artifact.
Uses MLflow's local file-based tracking store (`./mlruns`) by default — no MLflow
server needed for local/offline use, per the project's own scope.

Deliberately optional: `log_backtest_run` returns None (not an error) if `mlflow`
isn't importable, so the Streamlit app and pytest suite never hard-depend on it.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd


def log_backtest_run(results_df: pd.DataFrame, summary_df: pd.DataFrame, params: dict,
                      experiment_name: str = "supply_chain_backtesting", run_name: str | None = None,
                      tracking_uri: str | None = None) -> str | None:
    """Logs one backtest run's params/metrics/artifacts to MLflow. Returns the run_id,
    or None if MLflow is unavailable (import failure) — callers should treat that as
    "tracking skipped", not an error."""
    try:
        import mlflow
    except ImportError:
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        for _, row in summary_df.iterrows():
            metric_name = str(row["metric"]).replace(" ", "_")
            if pd.notna(row["mean"]):
                mlflow.log_metric(f"{metric_name}_mean", float(row["mean"]))
            if pd.notna(row["std"]):
                mlflow.log_metric(f"{metric_name}_std", float(row["std"]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            fold_path = os.path.join(tmp_dir, "fold_results.csv")
            results_df.to_csv(fold_path, index=False)
            mlflow.log_artifact(fold_path)

            plot_path = _calibration_plot(results_df, tmp_dir)
            if plot_path:
                mlflow.log_artifact(plot_path)

        return run.info.run_id


def _calibration_plot(results_df: pd.DataFrame, out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results_df["fold"], results_df["empirical_coverage"], marker="o", label="Empirical coverage")
    if "nominal_coverage" in results_df.columns:
        ax.axhline(results_df["nominal_coverage"].iloc[0], color="gray", linestyle="--",
                   label="Nominal target")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Coverage")
    ax.set_title("Rolling-origin calibration per fold")
    ax.legend()
    path = os.path.join(out_dir, "calibration_plot.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
