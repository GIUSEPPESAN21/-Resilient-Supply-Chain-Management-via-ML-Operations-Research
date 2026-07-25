"""Rolling-origin (walk-forward) cross-validation for the forecasters — the standard
time-series CV recipe (Hyndman & Athanasopoulos, "Forecasting: Principles and
Practice", ch. 5.10) adapted to this project's multi-customer panel data: train on
[0, cutoff], evaluate on (cutoff, cutoff + horizon], slide the cutoff forward, repeat.
Reports pinball loss / CRPS / empirical coverage per fold, aggregated as mean +/- std
across folds — never a single train/test split, which this project previously had
nothing better than.

Works with ANY forecaster exposing `.fit(features, target_col)` and
`.predict_quantiles(features) -> {quantile_level: array}` — both `XGBQuantileEnsemble`
and `ConformalizedQuantileForecaster` already share this interface (see
modules/forecasting.py), so the same backtester drives both without modification.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from modules.metrics import crps_from_quantiles, empirical_coverage, pinball_loss


def rolling_origin_folds(features: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5,
                          horizon_frac: float = 0.1) -> list[tuple[pd.DataFrame, pd.DataFrame, object, object]]:
    """Generates `n_folds` walk-forward folds using a GLOBAL date cutoff (shared
    across all customers) that slides forward in evenly-spaced steps between
    `min_train_frac` of history and the point that leaves room for one more
    horizon window. Returns (train_df, test_df, cutoff_date, horizon_end_date)
    tuples.
    """
    dates = sorted(features["date"].unique())
    n = len(dates)
    start_idx = int(n * min_train_frac)
    horizon = max(1, int(n * horizon_frac))
    max_cutoff_idx = n - horizon - 1
    if max_cutoff_idx <= start_idx:
        raise ValueError(f"Not enough history ({n} dates) for {n_folds} folds at "
                          f"min_train_frac={min_train_frac}, horizon_frac={horizon_frac}.")

    cutoff_indices = np.unique(np.linspace(start_idx, max_cutoff_idx, n_folds, dtype=int))
    folds = []
    for idx in cutoff_indices:
        cutoff_date = dates[idx]
        horizon_end_date = dates[min(idx + horizon, n - 1)]
        train_df = features[features["date"] <= cutoff_date]
        test_df = features[(features["date"] > cutoff_date) & (features["date"] <= horizon_end_date)]
        folds.append((train_df, test_df, cutoff_date, horizon_end_date))
    return folds


def rolling_origin_backtest(features: pd.DataFrame, forecaster_factory: Callable[[], object],
                             quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9), n_folds: int = 5,
                             min_train_frac: float = 0.5, horizon_frac: float = 0.1,
                             target_col: str = "demand") -> pd.DataFrame:
    """Runs the walk-forward backtest and returns a per-fold metrics DataFrame.
    `forecaster_factory` must return a FRESH, unfitted forecaster instance each
    call (so folds don't leak state) exposing `.fit(features, target_col)` and
    `.predict_quantiles(features)`.
    """
    folds = rolling_origin_folds(features, n_folds, min_train_frac, horizon_frac)
    lower_level, upper_level = min(quantile_levels), max(quantile_levels)

    rows = []
    for fold_i, (train_df, test_df, cutoff_date, horizon_end_date) in enumerate(folds):
        if train_df.empty or test_df.empty:
            continue
        model = forecaster_factory()
        model.fit(train_df, target_col)
        preds = model.predict_quantiles(test_df)
        y_true = test_df[target_col].to_numpy(dtype=float)

        row = {
            "fold": fold_i, "cutoff_date": cutoff_date, "horizon_end_date": horizon_end_date,
            "n_train": len(train_df), "n_test": len(test_df),
        }
        for q in quantile_levels:
            row[f"pinball_q{q}"] = pinball_loss(y_true, preds[q], q)
        row["crps"] = crps_from_quantiles(y_true, preds, quantile_levels)
        row["empirical_coverage"] = empirical_coverage(y_true, preds[lower_level], preds[upper_level])
        row["nominal_coverage"] = upper_level - lower_level
        rows.append(row)

    if not rows:
        raise RuntimeError("No fold produced train/test data — check min_train_frac/horizon_frac "
                            "against the available history.")
    return pd.DataFrame(rows)


def summarize_backtest(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across folds for each metric column — the aggregate view that
    matters, since any single fold's numbers are noise."""
    non_metric_cols = {"fold", "cutoff_date", "horizon_end_date", "n_train", "n_test"}
    metric_cols = [c for c in results_df.columns if c not in non_metric_cols]
    summary = results_df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return summary.reset_index().rename(columns={"index": "metric"})
