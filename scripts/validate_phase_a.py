"""Phase A acceptance-criteria validation: run this to get real numbers, not just
"it works". Checks:
  1. Quantile non-crossing after `enforce_monotonic_quantiles` (assertion).
  2. CQR empirical coverage vs. nominal target, on a genuinely held-out slice,
     across 3 different GVI scenario settings, compared against the naive
     (uncalibrated, independently-fit) baseline.

Usage:  .venv/Scripts/python.exe scripts/validate_phase_a.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.generate_data import load_or_generate_data
from modules.forecasting import (ConformalizedQuantileForecaster, XGBQuantileEnsemble,
                                  build_feature_frame, enforce_monotonic_quantiles,
                                  run_forecast_diagnostics, simulate_online_coverage,
                                  train_test_holdout_split)

XGB_KWARGS = dict(n_estimators=150, max_depth=4)  # "Fast" perf-mode settings, for speed
GVI_SCENARIOS = [20, 50, 80]
ALPHA = 0.1


def check_monotonicity(features) -> None:
    ensemble = XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS)
    ensemble.fit(features)
    raw = ensemble.predict_quantiles(features)
    n_crossed_before = int(np.sum((raw[0.1] > raw[0.5]) | (raw[0.5] > raw[0.9])))

    fixed = enforce_monotonic_quantiles(raw)
    violations = (fixed[0.1] > fixed[0.5]) | (fixed[0.5] > fixed[0.9])
    n_crossed_after = int(np.sum(violations))

    print(f"[A2] Quantile crossing before fix: {n_crossed_before}/{len(raw[0.5])} rows")
    print(f"[A2] Quantile crossing after  fix: {n_crossed_after}/{len(raw[0.5])} rows")
    assert n_crossed_after == 0, "enforce_monotonic_quantiles failed to remove all crossings"
    print("[A2] PASS — quantiles are provably non-crossing after the rearrangement fix.\n")


def check_coverage(demand, exogenous) -> None:
    print(f"[A1/A3] CQR vs naive empirical coverage (nominal target = {1 - ALPHA:.2f}), "
          f"held-out slice, across {len(GVI_SCENARIOS)} GVI scenarios:\n")
    header = f"{'GVI':>5} | {'n_holdout':>9} | {'naive cov':>9} | {'CQR cov':>8} | {'naive width':>11} | {'CQR width':>9}"
    print(header)
    print("-" * len(header))

    all_cqr_gaps = []
    for gvi in GVI_SCENARIOS:
        exo_scenario = exogenous.copy()
        exo_scenario.loc[exo_scenario.index[-30:], "gvi"] = gvi
        features = build_feature_frame(demand, exo_scenario)

        result = run_forecast_diagnostics(features, alpha=ALPHA, calib_frac=0.2,
                                           holdout_frac=0.15, **XGB_KWARGS)
        naive, cqr = result["naive"], result["cqr"]
        print(f"{gvi:>5} | {result['n_holdout']:>9} | {naive['empirical_coverage']:>9.3f} | "
              f"{cqr['empirical_coverage']:>8.3f} | {naive['interval_width']['mean']:>11.1f} | "
              f"{cqr['interval_width']['mean']:>9.1f}")
        all_cqr_gaps.append(abs(cqr["empirical_coverage"] - result["nominal_coverage"]))

    print()
    max_gap_pp = max(all_cqr_gaps) * 100
    print(f"[A1/A3] Max |CQR empirical coverage - nominal| across scenarios: {max_gap_pp:.1f} pp")
    if max_gap_pp <= 3.0:
        print("[A1/A3] PASS — within the ~2-3 pp acceptance band.")
    else:
        print("[A1/A3] MISS — exceeds the ~2-3 pp acceptance band. Reporting honestly, not hiding it.")


def check_adaptive_coverage(demand, exogenous, gvi: int = 20, alpha: float = ALPHA) -> None:
    """Sequential, date-ordered replay over the held-out slice comparing the static
    CQR correction against the online adaptive (Gibbs & Candes 2021) update, to see
    whether adaptivity actually closes a coverage gap caused by non-stationarity
    (trend/seasonality/GVI drift) that breaks split-conformal's exchangeability
    assumption."""
    exo_scenario = exogenous.copy()
    exo_scenario.loc[exo_scenario.index[-30:], "gvi"] = gvi
    features = build_feature_frame(demand, exo_scenario)
    train_df, holdout_df = train_test_holdout_split(features, holdout_frac=0.15)

    lower_level, upper_level = alpha / 2, 1 - alpha / 2
    cqr = ConformalizedQuantileForecaster(lower_quantile=lower_level, upper_quantile=upper_level,
                                          calib_frac=0.2, adaptive=True, adaptive_gamma=0.05,
                                          **XGB_KWARGS)
    cqr.fit(train_df)
    replay = simulate_online_coverage(cqr, holdout_df, alpha=alpha)

    n, nominal = replay["n"], replay["nominal_coverage"]
    print(f"\n[A1 stretch] Sequential date-ordered replay over holdout (n={n}, gvi={gvi}):")
    print(f"  static   CQR coverage: {replay['static_coverage']:.3f}  "
          f"(gap {abs(replay['static_coverage'] - nominal) * 100:.1f} pp)")
    print(f"  adaptive CQR coverage: {replay['adaptive_coverage']:.3f}  "
          f"(gap {abs(replay['adaptive_coverage'] - nominal) * 100:.1f} pp)")


if __name__ == "__main__":
    data = load_or_generate_data(n_customers=16, n_days=400, seed=42)
    demand, exogenous = data["demand"], data["exogenous"]
    features_base = build_feature_frame(demand, exogenous)

    check_monotonicity(features_base)
    check_coverage(demand, exogenous)
    check_adaptive_coverage(demand, exogenous, gvi=20)
    check_adaptive_coverage(demand, exogenous, gvi=50)
