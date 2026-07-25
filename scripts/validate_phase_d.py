"""Phase D acceptance-criteria validation.
Checks:
  1. Rolling-origin backtest runs end-to-end for both XGBQuantileEnsemble and
     ConformalizedQuantileForecaster, producing per-fold + mean/std metrics.
  2. compare_forecasters(): Diebold-Mariano for 2 forecasters, Friedman+Nemenyi for 3.
  3. MLflow logging (optional) actually writes a run when available.

Usage:  .venv/Scripts/python.exe scripts/validate_phase_d.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.generate_data import load_or_generate_data
from modules.backtesting import rolling_origin_backtest, summarize_backtest
from modules.forecasting import ConformalizedQuantileForecaster, XGBQuantileEnsemble, build_feature_frame
from modules.mlflow_tracking import log_backtest_run
from modules.significance import compare_forecasters

XGB_KWARGS = dict(n_estimators=100, max_depth=4)
N_FOLDS = 5


class NaiveRepeatLastForecaster:
    """Trivial 'repeat yesterday's demand for every quantile' baseline — used only in
    this validation script to exercise the >2-forecaster Friedman/Nemenyi path with a
    genuinely different (and genuinely worse) third method."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        self.quantiles = quantiles

    def fit(self, features, target_col="demand"):
        return self

    def predict_quantiles(self, features):
        base = features["lag_1"].to_numpy(dtype=float)
        return {q: base.copy() for q in self.quantiles}


def check_backtesting(features) -> dict:
    print("[D1] Rolling-origin backtest:")
    xgb_results = rolling_origin_backtest(
        features, lambda: XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS),
        n_folds=N_FOLDS, min_train_frac=0.5, horizon_frac=0.1)
    xgb_summary = summarize_backtest(xgb_results)
    print("  XGBQuantileEnsemble per-fold CRPS:", xgb_results["crps"].round(3).tolist())
    print("  XGBQuantileEnsemble summary (mean +/- std):")
    print(xgb_summary.to_string(index=False))

    cqr_results = rolling_origin_backtest(
        features, lambda: ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9,
                                                           calib_frac=0.2, **XGB_KWARGS),
        n_folds=N_FOLDS, min_train_frac=0.5, horizon_frac=0.1)
    cqr_summary = summarize_backtest(cqr_results)
    print("\n  ConformalizedQuantileForecaster (uncalibrated predict_quantiles) per-fold CRPS:",
          cqr_results["crps"].round(3).tolist())
    print("  ConformalizedQuantileForecaster summary (mean +/- std):")
    print(cqr_summary.to_string(index=False))

    assert len(xgb_results) >= 3, "Too few folds produced — check min_train_frac/horizon_frac"
    print(f"\n[D1] PASS — {len(xgb_results)} folds produced, per-fold metrics computed, "
          f"aggregated as mean +/- std.\n")
    return {"xgb_results": xgb_results, "xgb_summary": xgb_summary,
            "cqr_results": cqr_results, "cqr_summary": cqr_summary}


def check_significance(features, xgb_results, cqr_results) -> None:
    print("[D2] Statistical significance testing:")
    dm = compare_forecasters({"XGBQuantileEnsemble": xgb_results["crps"].to_numpy(),
                               "ConformalizedQuantileForecaster": cqr_results["crps"].to_numpy()})
    print(f"  Diebold-Mariano (XGB vs CQR, per-fold CRPS): stat={dm['dm_stat']:.3f}, "
          f"p={dm['p_value']:.4f}, significant@0.05={dm['significant']}")

    naive_results = rolling_origin_backtest(
        features, lambda: NaiveRepeatLastForecaster(), n_folds=N_FOLDS, min_train_frac=0.5,
        horizon_frac=0.1)
    print(f"  NaiveRepeatLastForecaster per-fold CRPS: {naive_results['crps'].round(3).tolist()}")

    fn = compare_forecasters({
        "XGBQuantileEnsemble": xgb_results["crps"].to_numpy(),
        "ConformalizedQuantileForecaster": cqr_results["crps"].to_numpy(),
        "NaiveRepeatLast": naive_results["crps"].to_numpy(),
    })
    print(f"  Friedman (3 forecasters, per-fold CRPS): stat={fn['friedman_stat']:.3f}, "
          f"p={fn['p_value']:.4f}, significant@0.05={fn['significant']}")
    print(f"  Average ranks (lower=better): {fn['avg_ranks']}")
    print(f"  Nemenyi critical difference: {fn['critical_difference']:.3f}")
    print("[D2] PASS — both DM (2-way) and Friedman/Nemenyi (3-way) run without error.\n")


def check_mlflow(xgb_results, xgb_summary) -> None:
    print("[D3] MLflow experiment tracking:")
    run_id = log_backtest_run(xgb_results, xgb_summary,
                               params={"model": "XGBQuantileEnsemble", "n_estimators": XGB_KWARGS["n_estimators"],
                                       "max_depth": XGB_KWARGS["max_depth"], "n_folds": N_FOLDS},
                               experiment_name="phase_d_validation")
    if run_id is None:
        print("[D3] MLflow not importable in this environment — tracking skipped gracefully "
              "(log_backtest_run returned None, not an error).\n")
    else:
        print(f"  Logged MLflow run_id={run_id} (local tracking store, ./mlruns)")
        print("[D3] PASS — backtest run logged to MLflow.\n")


if __name__ == "__main__":
    data = load_or_generate_data(n_customers=16, n_days=400, seed=42)
    demand, exogenous = data["demand"], data["exogenous"]
    features = build_feature_frame(demand, exogenous)

    bt = check_backtesting(features)
    check_significance(features, bt["xgb_results"], bt["cqr_results"])
    check_mlflow(bt["xgb_results"], bt["xgb_summary"])
