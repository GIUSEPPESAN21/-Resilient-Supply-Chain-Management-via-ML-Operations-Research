"""One-command rolling-origin backtest runner.

Usage:
    .venv/Scripts/python.exe scripts/run_backtest.py [--model xgb|cqr] [--customers N]
        [--days N] [--folds N] [--mlflow] [--out-dir DIR]

Produces a per-fold metrics table (printed + saved to CSV), a mean +/- std summary,
and a calibration plot (PNG) — the single command referenced by Phase D's acceptance
criteria ("a rolling-origin backtest can be run end-to-end via one command").
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import load_or_generate_data
from modules.backtesting import rolling_origin_backtest, summarize_backtest
from modules.forecasting import ConformalizedQuantileForecaster, XGBQuantileEnsemble, build_feature_frame
from modules.mlflow_tracking import log_backtest_run
from modules.mlflow_tracking import _calibration_plot as calibration_plot


def build_factory(model: str, n_estimators: int):
    if model == "xgb":
        return lambda: XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), n_estimators=n_estimators)
    return lambda: ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9,
                                                    calib_frac=0.2, n_estimators=n_estimators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["xgb", "cqr"], default="cqr")
    parser.add_argument("--customers", type=int, default=16)
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--mlflow", action="store_true", help="Also log this run to MLflow.")
    parser.add_argument("--out-dir", default="backtest_output")
    args = parser.parse_args()

    data = load_or_generate_data(n_customers=args.customers, n_days=args.days, seed=42)
    features = build_feature_frame(data["demand"], data["exogenous"])
    factory = build_factory(args.model, args.n_estimators)

    results = rolling_origin_backtest(features, factory, n_folds=args.folds)
    summary = summarize_backtest(results)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "fold_results.csv")
    results.to_csv(results_path, index=False)

    print(f"Model: {args.model} | customers={args.customers} days={args.days} folds={len(results)}\n")
    print("Per-fold metrics:")
    print(results.to_string(index=False))
    print("\nSummary (mean +/- std across folds):")
    print(summary.to_string(index=False))
    print(f"\nPer-fold results saved to: {results_path}")

    plot_path = calibration_plot(results, args.out_dir)
    if plot_path:
        print(f"Calibration plot saved to: {plot_path}")

    if args.mlflow:
        run_id = log_backtest_run(results, summary,
                                   params={"model": args.model, "n_folds": args.folds,
                                           "n_customers": args.customers, "n_days": args.days,
                                           "n_estimators": args.n_estimators})
        print(f"MLflow run_id: {run_id}" if run_id else
              "MLflow not available in this environment — tracking skipped.")


if __name__ == "__main__":
    main()
