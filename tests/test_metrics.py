import numpy as np
import pytest

from modules.metrics import crps_from_quantiles, empirical_coverage, interval_width, pinball_loss


def test_pinball_loss_median_matches_half_absolute_error():
    # q=0.5 pinball loss reduces to 0.5 * |y - pred|, hand-computable directly.
    y_true = [10, 20, 30]
    y_pred = [15, 15, 15]
    # diffs: -5, 5, 15 -> 0.5*|diff| = 2.5, 2.5, 7.5 -> mean = 4.1666...
    assert pinball_loss(y_true, y_pred, quantile=0.5) == pytest.approx(12.5 / 3, abs=1e-9)


def test_pinball_loss_asymmetric_quantile_hand_computed():
    # single observation, quantile=0.1: diff = y - pred = 10 - 5 = 5 (under-prediction)
    # loss = max(q*diff, (q-1)*diff) = max(0.1*5, -0.9*5) = max(0.5, -4.5) = 0.5
    assert pinball_loss([10], [5], quantile=0.1) == pytest.approx(0.5)
    # over-prediction: diff = 10 - 15 = -5 -> loss = max(0.1*-5, -0.9*-5) = max(-0.5, 4.5) = 4.5
    assert pinball_loss([10], [15], quantile=0.1) == pytest.approx(4.5)


def test_crps_from_quantiles_hand_computed_single_observation():
    # y=10, quantile preds q0.1=5 (loss 0.5), q0.5=10 (loss 0), q0.9=15 (loss 0.5) --
    # see test_pinball_loss_* above for the underlying per-quantile loss values.
    # Trapezoidal integration of [0.5, 0, 0.5] over levels [0.1, 0.5, 0.9]:
    #   (0.1,0.5) segment: avg(0.5,0)*0.4 = 0.1 ; (0.5,0.9) segment: avg(0,0.5)*0.4 = 0.1
    #   integral = 0.2 ; CRPS = 2 * 0.2 = 0.4
    crps = crps_from_quantiles([10], {0.1: [5], 0.5: [10], 0.9: [15]}, [0.1, 0.5, 0.9])
    assert crps == pytest.approx(0.4, abs=1e-9)


def test_crps_from_quantiles_perfect_forecast_is_zero():
    y_true = [10, 20, 30]
    preds = {0.1: y_true, 0.5: y_true, 0.9: y_true}
    assert crps_from_quantiles(y_true, preds, [0.1, 0.5, 0.9]) == pytest.approx(0.0, abs=1e-9)


def test_empirical_coverage_hand_computed():
    y_true = [5, 15, 35]
    lower = [0, 10, 20]
    upper = [10, 20, 30]
    # 5 in [0,10]: yes. 15 in [10,20]: yes. 35 in [20,30]: no. -> 2/3
    assert empirical_coverage(y_true, lower, upper) == pytest.approx(2 / 3)


def test_empirical_coverage_all_covered_and_none_covered():
    assert empirical_coverage([5], [0], [10]) == 1.0
    assert empirical_coverage([50], [0], [10]) == 0.0


def test_interval_width_hand_computed():
    lower = [0, 10, 20]
    upper = [10, 25, 20]
    result = interval_width(lower, upper)
    # widths: 10, 15, 0
    assert result["mean"] == pytest.approx(25 / 3)
    assert result["min"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(15.0)


def test_interval_width_zero_when_lower_equals_upper():
    result = interval_width([1, 2, 3], [1, 2, 3])
    assert result["mean"] == 0.0
    assert result["max"] == 0.0
