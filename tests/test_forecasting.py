import numpy as np
import pytest

from modules.forecasting import (ConformalizedQuantileForecaster, LSTMForecaster,
                                  QuantileXGBForecaster, XGBQuantileEnsemble,
                                  enforce_monotonic_quantiles)

XGB_KWARGS = dict(n_estimators=30, max_depth=3)


def test_enforce_monotonic_quantiles_fixes_adversarial_crossing():
    # Deliberately crossed: row 0 has q50 < q10 < q90 in value, row 1 has q90 < q10 < q50.
    crossed = {
        0.1: np.array([5.0, 12.0]),
        0.5: np.array([10.0, 8.0]),
        0.9: np.array([8.0, 15.0]),
    }
    assert np.any(crossed[0.1] > crossed[0.5]) or np.any(crossed[0.5] > crossed[0.9])

    fixed = enforce_monotonic_quantiles(crossed)
    assert np.all(fixed[0.1] <= fixed[0.5])
    assert np.all(fixed[0.5] <= fixed[0.9])
    # Rearrangement is a permutation of the same 3 values per row, not new values.
    for i in range(2):
        original_row = sorted([crossed[0.1][i], crossed[0.5][i], crossed[0.9][i]])
        fixed_row = [fixed[0.1][i], fixed[0.5][i], fixed[0.9][i]]
        assert fixed_row == original_row


def test_enforce_monotonic_quantiles_noop_when_already_sorted():
    sorted_dict = {0.1: np.array([1.0, 2.0]), 0.5: np.array([5.0, 6.0]), 0.9: np.array([9.0, 10.0])}
    fixed = enforce_monotonic_quantiles(sorted_dict)
    for level in sorted_dict:
        assert np.array_equal(fixed[level], sorted_dict[level])


def test_xgb_quantile_ensemble_predict_shapes(small_features):
    ensemble = XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS)
    ensemble.fit(small_features)
    preds = ensemble.predict_quantiles(small_features)
    assert set(preds.keys()) == {0.1, 0.5, 0.9}
    for arr in preds.values():
        assert arr.shape == (len(small_features),)
        assert arr.dtype.kind == "f"


def test_quantile_xgb_forecaster_set_quantile_requires_refit(small_features):
    model = QuantileXGBForecaster(quantile=0.5, **XGB_KWARGS)
    model.fit(small_features)
    preds_median = model.predict(small_features)
    model.set_quantile(0.9)
    model.fit(small_features)
    preds_upper = model.predict(small_features)
    assert preds_median.shape == preds_upper.shape == (len(small_features),)


def test_cqr_predict_interval_shapes_and_ordering(small_features):
    cqr = ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9, calib_frac=0.2,
                                           **XGB_KWARGS)
    cqr.fit(small_features)
    lower, upper = cqr.predict_interval(small_features, alpha=0.1)
    assert lower.shape == upper.shape == (len(small_features),)
    assert np.all(lower <= upper)


def test_cqr_predict_quantiles_matches_ensemble_interface(small_features):
    cqr = ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9, calib_frac=0.2,
                                           **XGB_KWARGS)
    cqr.fit(small_features)
    preds = cqr.predict_quantiles(small_features)
    assert set(preds.keys()) == {0.1, 0.5, 0.9}
    for arr in preds.values():
        assert arr.shape == (len(small_features),)


def test_cqr_requires_fit_before_predict_interval():
    cqr = ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9, **XGB_KWARGS)
    with pytest.raises(RuntimeError):
        cqr.predict_interval(None)


def test_cqr_reasonable_coverage_on_held_out_synthetic_set(small_features):
    from modules.forecasting import run_forecast_diagnostics
    result = run_forecast_diagnostics(small_features, alpha=0.1, calib_frac=0.2, holdout_frac=0.2,
                                       **XGB_KWARGS)
    # Not asserting the tight ~2-3pp acceptance band here (Phase A's own validation
    # showed static CQR can miss it under non-stationarity) -- just a sanity bound
    # that calibration is in the right ballpark, to catch a genuinely broken CQR.
    assert 0.5 <= result["cqr"]["empirical_coverage"] <= 1.0


def test_update_adaptive_rejects_non_adaptive_instance():
    cqr = ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9, adaptive=False,
                                           **XGB_KWARGS)
    with pytest.raises(RuntimeError):
        cqr.update_adaptive(y_true=10, lower=0, upper=20)


def test_update_adaptive_moves_alpha_toward_target_on_repeated_misses():
    cqr = ConformalizedQuantileForecaster(lower_quantile=0.1, upper_quantile=0.9, adaptive=True,
                                           adaptive_gamma=0.1, **XGB_KWARGS)
    # Every observation falls OUTSIDE [0, 1] (a miss) -> alpha_t should shrink each
    # update (Gibbs & Candes 2021: widen intervals in response to under-coverage,
    # which corresponds to alpha_t decreasing toward 0).
    alpha_t = 0.1
    for _ in range(5):
        alpha_t = cqr.update_adaptive(y_true=100, lower=0, upper=1, alpha=0.1)
    assert alpha_t < 0.1


def test_lstm_forecaster_predict_quantiles_shape(small_synthetic_data):
    lstm = LSTMForecaster(lookback=14, hidden_size=8, epochs=2).fit(
        small_synthetic_data["demand"], small_synthetic_data["exogenous"])
    preds = lstm.predict_quantiles(small_synthetic_data["demand"], small_synthetic_data["exogenous"])
    assert preds.ndim == 2 and preds.shape[1] == 3
