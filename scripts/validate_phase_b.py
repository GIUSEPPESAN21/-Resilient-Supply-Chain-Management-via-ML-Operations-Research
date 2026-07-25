"""Phase B acceptance-criteria validation: run this to get real numbers.
Checks:
  1. Existing deterministic solve_cvrp_sd still works unmodified (regression).
  2. SAA mode's service_level_achieved is in [0,1] and, after capacity-building +
     a FRESH independent Monte Carlo re-validation, lands within a few points of
     the target alpha.
  3. VSS and EVPI are computed without error; reported honestly whether >= 0 or not.
  4. OSRM distance backend: real endpoint if reachable, else documented fallback.

Usage:  .venv/Scripts/python.exe scripts/validate_phase_b.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.generate_data import load_or_generate_data
from modules.forecasting import ConformalizedQuantileForecaster, build_feature_frame
from modules.optimization import solve_cvrp_sd, suggest_fleet_size
from modules.optimization_saa import (sample_demand_scenarios, solve_saa_cvrp_sd,
                                       solve_saa_cvrp_sd_target_capacity)
from modules.routing_backend import get_distance_matrix
from modules.value_of_information import compute_vss_evpi_report

XGB_KWARGS = dict(n_estimators=150, max_depth=4)
VEHICLE_CAPACITY = 500
ALPHA = 0.1  # CQR miscoverage -> 90% calibrated interval feeds the demand scenarios
SERVICE_LEVEL_ALPHA = 0.95


def check_deterministic_regression(depot, customers, demand_point) -> dict:
    num_vehicles = suggest_fleet_size(demand_point, VEHICLE_CAPACITY)
    result = solve_cvrp_sd(depot, customers, demand_point, VEHICLE_CAPACITY, num_vehicles, time_limit_s=3)
    print(f"[B regression] Existing solve_cvrp_sd: feasible={result['feasible']}, "
          f"total_distance_km={result['total_distance_km']}, "
          f"vehicles_used={result['num_vehicles_used']}")
    assert result["feasible"], "Existing deterministic solver regressed!"
    print("[B regression] PASS — unmodified existing solver still works.\n")
    return result


def check_saa(depot, customers, median, lower, upper, num_vehicles) -> dict:
    print(f"[B1] SAA CVRP-SD, target service level alpha={SERVICE_LEVEL_ALPHA}, "
          f"nominal capacity={VEHICLE_CAPACITY}:")
    result = solve_saa_cvrp_sd_target_capacity(
        depot, customers, median, lower, upper, ALPHA, VEHICLE_CAPACITY, num_vehicles,
        time_limit_s=3, n_scenarios=300, service_level_alpha=SERVICE_LEVEL_ALPHA)

    assert result["feasible"], "SAA solve came back infeasible"
    achieved = result["overall_service_level_achieved"]
    assert 0.0 <= achieved <= 1.0, f"service_level_achieved out of [0,1]: {achieved}"

    print(f"  capacity_adjusted: {result['capacity_adjusted']}  "
          f"(capacity used: {result['capacity_used']})")
    print(f"  overall_service_level_achieved: {achieved:.3f}  (target {SERVICE_LEVEL_ALPHA})")
    print(f"  mean_route_service_level:       {result['mean_route_service_level']:.3f}")
    print(f"  expected_recourse_cost:         {result['expected_recourse_cost']:.2f}")
    gap_pp = abs(achieved - SERVICE_LEVEL_ALPHA) * 100
    print(f"  gap vs. target: {gap_pp:.1f} pp")
    if gap_pp <= 5.0:
        print("[B1] PASS — SAA-adjusted plan's re-validated service level is close to target.\n")
    else:
        print("[B1] MISS — re-validated service level misses target by more than a few points. "
              "Reporting honestly.\n")
    return result


def check_vss_evpi(depot, customers, median, lower, upper, num_vehicles) -> dict:
    """VSS/EVPI requires a FAIR comparison: EV and RP must compete for the SAME
    vehicle_capacity, differing only in what demand information the routing decision
    used (mean vs. a robust quantile statistic) -- not in how much capacity each got.
    So this deliberately uses the BASE `solve_saa_cvrp_sd` (fixed nominal capacity),
    not the capacity-building wrapper from check_saa (which is a separate, valid
    feature for sizing capacity to a service-level target, but would confound "value
    of stochastic information" with "value of extra capacity" if used here.
    """
    print("[B3] VSS / EVPI report:")
    scenarios = sample_demand_scenarios(median, lower, upper, ALPHA, n_scenarios=300, random_state=123)
    saa_plan = solve_saa_cvrp_sd(depot, customers, median, lower, upper, ALPHA, VEHICLE_CAPACITY,
                                  num_vehicles, time_limit_s=3, n_scenarios=300,
                                  service_level_alpha=SERVICE_LEVEL_ALPHA, random_state=99)
    assert saa_plan["feasible"], "Base SAA solve (fixed capacity) came back infeasible"

    report = compute_vss_evpi_report(depot, customers, scenarios, saa_plan, VEHICLE_CAPACITY,
                                      num_vehicles, time_limit_s=3, n_scenarios_ws=20,
                                      ws_time_limit_s=1)
    ev_plan = report["deterministic_plan"]["plan"]
    ev_eval = report["deterministic_plan"]["evaluation"]
    rp_eval = report["stochastic_evaluation"]["evaluation"]
    print(f"  EV plan:  distance={ev_plan['total_distance_km']:.2f} km, "
          f"vehicles={ev_plan['num_vehicles_used']}, "
          f"E[recourse]={ev_eval['expected_recourse_cost']:.2f}, "
          f"service_level={ev_eval['overall_service_level_achieved']:.3f}")
    print(f"  RP plan:  distance={saa_plan['total_distance_km']:.2f} km, "
          f"vehicles={saa_plan['num_vehicles_used']}, "
          f"E[recourse]={rp_eval['expected_recourse_cost']:.2f}, "
          f"service_level={rp_eval['overall_service_level_achieved']:.3f}, "
          f"capacity={VEHICLE_CAPACITY} (same as EV)")
    print(f"  EEV cost (deterministic mean-demand plan, evaluated stochastically): {report['EEV_cost']:.2f}")
    print(f"  RP  cost (SAA/stochastic plan, evaluated stochastically):           {report['RP_cost']:.2f}")
    print(f"  WS  cost (wait-and-see, perfect foresight, {report['wait_and_see']['n_used']} scenarios): "
          f"{report['WS_cost']:.2f}")
    print(f"  VSS  = EEV - RP  = {report['VSS']:.2f}  ({'>= 0, as expected' if report['VSS'] >= 0 else 'NEGATIVE -- reporting honestly'})")
    print(f"  EVPI = RP  - WS  = {report['EVPI']:.2f}  ({'>= 0, as expected' if report['EVPI'] >= 0 else 'NEGATIVE -- reporting honestly'})")
    print()
    return report


def check_osrm_fallback() -> None:
    print("[B4] OSRM distance backend:")
    lats = np.array([4.7110, 4.72, 4.70, 4.715])
    lons = np.array([-74.0721, -74.06, -74.08, -74.07])

    dist_km, mode_used, warning = get_distance_matrix(lats, lons, mode="osrm", timeout=4.0)
    print(f"  requested mode=osrm -> mode_used={mode_used}")
    if warning:
        print(f"  fallback triggered: {warning}")
    else:
        print(f"  OSRM reachable — sample distance[0,1] = {dist_km[0, 1]:.3f} km")

    dist_km_bad, mode_used_bad, warning_bad = get_distance_matrix(
        lats, lons, mode="osrm", osrm_base_url="http://localhost:1", timeout=1.0)
    print(f"  deliberately-unreachable OSRM base_url -> mode_used={mode_used_bad}")
    assert mode_used_bad == "haversine" and warning_bad is not None, \
        "OSRM failure did not fall back to haversine with a warning!"
    print("[B4] PASS — deliberate OSRM failure falls back to haversine with an explicit warning, "
          "no crash, no silent wrong distances.\n")


if __name__ == "__main__":
    data = load_or_generate_data(n_customers=16, n_days=400, seed=42)
    demand, exogenous, customers, depot = data["demand"], data["exogenous"], data["customers"], data["depot"]
    features = build_feature_frame(demand, exogenous)

    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].sort_values("customer_id")
    check_deterministic_regression(depot, customers, latest["q_target"].to_numpy() if "q_target" in latest.columns else latest["demand"].to_numpy())

    lower_level, upper_level = ALPHA / 2, 1 - ALPHA / 2
    cqr = ConformalizedQuantileForecaster(lower_quantile=lower_level, upper_quantile=upper_level,
                                          calib_frac=0.2, **XGB_KWARGS)
    cqr.fit(features)
    lower, upper = cqr.predict_interval(latest, alpha=ALPHA)
    median = cqr.model_median.predict(latest)
    num_vehicles = suggest_fleet_size(median, VEHICLE_CAPACITY)

    check_saa(depot, customers, median, lower, upper, num_vehicles)
    check_vss_evpi(depot, customers, median, lower, upper, num_vehicles)
    check_osrm_fallback()
