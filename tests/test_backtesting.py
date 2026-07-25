import pytest

from modules.backtesting import rolling_origin_backtest, rolling_origin_folds, summarize_backtest
from modules.forecasting import XGBQuantileEnsemble

XGB_KWARGS = dict(n_estimators=30, max_depth=3)


def test_rolling_origin_folds_are_chronological_and_non_overlapping(small_features):
    folds = rolling_origin_folds(small_features, n_folds=3, min_train_frac=0.5, horizon_frac=0.1)
    assert len(folds) >= 2
    for train_df, test_df, cutoff_date, horizon_end_date in folds:
        assert train_df["date"].max() <= cutoff_date
        assert test_df["date"].min() > cutoff_date
        assert test_df["date"].max() <= horizon_end_date


def test_rolling_origin_folds_raises_when_history_too_short(small_features):
    with pytest.raises(ValueError):
        rolling_origin_folds(small_features, n_folds=3, min_train_frac=0.95, horizon_frac=0.3)


def test_rolling_origin_backtest_produces_per_fold_metrics(small_features):
    results = rolling_origin_backtest(
        small_features, lambda: XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS),
        n_folds=3, min_train_frac=0.5, horizon_frac=0.15)
    assert len(results) >= 2
    for col in ("pinball_q0.1", "pinball_q0.5", "pinball_q0.9", "crps", "empirical_coverage"):
        assert col in results.columns
        assert results[col].notna().all()


def test_summarize_backtest_reports_mean_and_std(small_features):
    results = rolling_origin_backtest(
        small_features, lambda: XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS),
        n_folds=3, min_train_frac=0.5, horizon_frac=0.15)
    summary = summarize_backtest(results)
    assert {"metric", "mean", "std"}.issubset(summary.columns)
    assert "crps" in summary["metric"].to_numpy()
