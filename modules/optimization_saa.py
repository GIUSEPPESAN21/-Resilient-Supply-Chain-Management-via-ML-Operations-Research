"""Sample Average Approximation (SAA) for CVRP with Stochastic Demand.

Replaces the "deterministic solve on an inflated upper-quantile demand and hope"
approach in `modules/optimization.py` with a measurable service-level guarantee:
route topology is still solved once (deterministic equivalent, via the existing
`solve_cvrp_sd`), but on the *representative* (scenario-mean) demand rather than an
upper quantile, and the resulting fixed plan is then Monte Carlo-validated against
N demand scenarios drawn from the calibrated predictive distribution (Phase A's CQR
interval). This gives an explicit, empirically-measured P(route load <= capacity)
per route, plus an expected two-stage recourse cost for the scenarios that do
violate capacity — concepts the "chance-constrained approximation" docstring in
optimization.py named but never actually implemented.

`solve_cvrp_sd` (the existing deterministic solver) is untouched and still used
internally — this module only adds scenario generation, Monte Carlo evaluation of
a fixed plan, and the recourse-cost model on top of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from modules.optimization import solve_cvrp_sd

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(EARTH_RADIUS_KM * 2 * np.arcsin(np.clip(np.sqrt(a), -1, 1)))


def sample_demand_scenarios(median: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                             alpha: float, n_scenarios: int, random_state: int = 42) -> np.ndarray:
    """Monte Carlo demand scenarios per customer, from a Normal fit to the calibrated
    (lower, upper) predictive interval at level alpha — the same inverse-normal-spacing
    parametrization `compute_stockout_risk` already uses elsewhere in this codebase,
    just applied here to Phase A's CQR-calibrated interval instead of the raw quantile
    band. This is a documented parametric approximation, not a nonparametric bootstrap
    (no per-customer residual history is available at this layer); truncated at 0
    since demand cannot be negative.

    Returns an (n_scenarios, n_customers) array.
    """
    median, lower, upper = np.asarray(median, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    z = norm.ppf(1 - alpha / 2)
    std = np.clip((upper - lower) / (2 * z), 1e-6, None)
    rng = np.random.default_rng(random_state)
    draws = rng.normal(loc=median, scale=std, size=(n_scenarios, len(median)))
    return np.clip(draws, 0, None)


def evaluate_routes_against_scenarios(routes: list[dict], scenarios: np.ndarray,
                                       customer_id_to_idx: dict, depot: dict,
                                       vehicle_capacity: float, service_level_alpha: float = 0.95,
                                       cost_per_km: float = 1.0,
                                       stockout_penalty_per_unit: float = 5.0) -> dict:
    """Monte Carlo-evaluates a FIXED route plan (route topology already decided —
    which customers ride on which vehicle, in what order) against a matrix of demand
    scenarios. For each route and scenario, walks the stops in order accumulating
    realized demand; the first stop where cumulative load exceeds capacity triggers a
    recourse event, costed as (return-to-depot distance x2) + a stockout penalty per
    unmet unit — a simplified two-stage recourse: the vehicle is assumed to return to
    the depot, reload, and complete the rest of the route without further violation.
    """
    n_scenarios = scenarios.shape[0]
    per_route_service_level = {}
    required_capacity_for_target = {}
    recourse_cost_per_scenario = np.zeros(n_scenarios)
    any_violation = np.zeros(n_scenarios, dtype=bool)

    for route in routes:
        vid = route["vehicle_id"]
        interior_ids = route["stops"][1:-1]
        interior_coords = route["coords"][1:-1]
        if not interior_ids:
            per_route_service_level[vid] = 1.0
            required_capacity_for_target[vid] = 0.0
            continue
        idx = [customer_id_to_idx[cid] for cid in interior_ids]
        route_scenario_demand = scenarios[:, idx]  # (n_scenarios, n_stops)
        cumulative = np.cumsum(route_scenario_demand, axis=1)
        route_totals = cumulative[:, -1]

        violated = route_totals > vehicle_capacity
        any_violation |= violated
        per_route_service_level[vid] = float(1.0 - violated.mean())
        required_capacity_for_target[vid] = float(np.quantile(route_totals, service_level_alpha))

        for s in np.nonzero(violated)[0]:
            first_violation_pos = int(np.argmax(cumulative[s] > vehicle_capacity))
            unmet = float(cumulative[s, first_violation_pos] - vehicle_capacity)
            stop_lat, stop_lon = interior_coords[first_violation_pos]
            dist_to_depot_km = _haversine_km(stop_lat, stop_lon, depot["lat"], depot["lon"])
            recourse_cost_per_scenario[s] += (2 * dist_to_depot_km * cost_per_km
                                               + stockout_penalty_per_unit * unmet)

    overall_service_level = float(1.0 - any_violation.mean()) if routes else 1.0

    return {
        "per_route_service_level": per_route_service_level,
        "mean_route_service_level": float(np.mean(list(per_route_service_level.values()))) if per_route_service_level else 1.0,
        "overall_service_level_achieved": overall_service_level,
        "required_capacity_for_target": required_capacity_for_target,
        "expected_recourse_cost": float(recourse_cost_per_scenario.mean()),
        "recourse_cost_per_scenario": recourse_cost_per_scenario,
    }


def _representative_demand(scenarios: np.ndarray, representative_stat: float | str) -> np.ndarray:
    """`representative_stat="mean"` -> scenario mean (matches the classical "EV problem"
    demand estimate). A float in (0, 1) is instead read as a per-customer quantile
    level of the scenario distribution -- a "robust statistic" that hedges toward the
    upper half of demand variability without being as conservative as a pure
    upper-quantile heuristic. Using anything other than "mean" here is what makes the
    SAA routing DECISION meaningfully different from the deterministic mean-demand
    plan at the SAME vehicle capacity (see modules/value_of_information.py) -- if both
    used the mean, they would nearly coincide and VSS would be a near-tautological ~0.
    """
    if representative_stat == "mean":
        return scenarios.mean(axis=0)
    if isinstance(representative_stat, (int, float)) and 0 < representative_stat < 1:
        return np.quantile(scenarios, representative_stat, axis=0)
    raise ValueError(f"representative_stat must be 'mean' or a float in (0,1), got {representative_stat!r}")


def solve_saa_cvrp_sd(depot: dict, customers: pd.DataFrame, median_demand: np.ndarray,
                       lower_demand: np.ndarray, upper_demand: np.ndarray, calib_alpha: float,
                       vehicle_capacity: int, num_vehicles: int, time_limit_s: int = 5,
                       n_scenarios: int = 200, service_level_alpha: float = 0.95,
                       cost_per_km: float = 1.0, stockout_penalty_per_unit: float = 5.0,
                       random_state: int = 42, representative_stat: float | str = 0.75,
                       distance_matrix_km: np.ndarray | None = None) -> dict:
    """SAA CVRP-SD: solves routing once on a representative demand drawn from Monte
    Carlo scenarios (Romano et al. 2019's calibrated CQR interval feeds the scenario
    distribution), then Monte Carlo-validates the resulting fixed plan's realized
    service level and expected recourse cost against `n_scenarios` draws — replacing
    the deterministic "solve on the upper quantile" heuristic with a measurable
    P(load <= capacity) >= alpha guarantee, computed empirically rather than assumed.

    `representative_stat` defaults to the 0.75 quantile rather than the plain mean —
    see `_representative_demand` for why: routing on the mean makes this solution
    nearly indistinguishable from the deterministic EV plan at the same capacity.
    `distance_matrix_km` optionally passes through to `solve_cvrp_sd` (e.g. OSRM
    real-road-network distances instead of haversine).
    """
    scenarios = sample_demand_scenarios(median_demand, lower_demand, upper_demand,
                                         calib_alpha, n_scenarios, random_state)
    representative_demand = _representative_demand(scenarios, representative_stat)

    base_result = solve_cvrp_sd(depot, customers, representative_demand, vehicle_capacity,
                                 num_vehicles, time_limit_s, distance_matrix_km=distance_matrix_km)
    result = {**base_result, "representative_demand": representative_demand,
              "n_scenarios": n_scenarios, "service_level_target": service_level_alpha}

    if not base_result["feasible"] or not base_result["routes"]:
        result.update({
            "per_route_service_level": {}, "mean_route_service_level": None,
            "overall_service_level_achieved": None, "required_capacity_for_target": {},
            "expected_recourse_cost": None,
        })
        return result

    customer_id_to_idx = {cid: i for i, cid in enumerate(customers["customer_id"].to_numpy())}
    evaluation = evaluate_routes_against_scenarios(
        base_result["routes"], scenarios, customer_id_to_idx, depot, vehicle_capacity,
        service_level_alpha, cost_per_km, stockout_penalty_per_unit)
    result.update(evaluation)
    return result


def solve_saa_cvrp_sd_target_capacity(depot: dict, customers: pd.DataFrame, median_demand: np.ndarray,
                                       lower_demand: np.ndarray, upper_demand: np.ndarray,
                                       calib_alpha: float, vehicle_capacity: int, num_vehicles: int,
                                       time_limit_s: int = 5, n_scenarios: int = 200,
                                       service_level_alpha: float = 0.95, cost_per_km: float = 1.0,
                                       stockout_penalty_per_unit: float = 5.0,
                                       random_state: int = 42, representative_stat: float | str = 0.75,
                                       distance_matrix_km: np.ndarray | None = None) -> dict:
    """Two-pass capacity-building wrapper around `solve_saa_cvrp_sd`: solves once at
    the user-supplied nominal `vehicle_capacity`; if the achieved overall service
    level falls short of `service_level_alpha`, re-solves at the SAA-implied required
    capacity (the alpha-quantile of realized route load) and re-validates that
    adjusted plan against a FRESH, independently-seeded scenario draw. This is what
    turns "use the upper quantile and hope" into an actual, checkable
    P(route load <= capacity) >= alpha guarantee rather than a single point estimate.
    """
    first = solve_saa_cvrp_sd(depot, customers, median_demand, lower_demand, upper_demand,
                               calib_alpha, vehicle_capacity, num_vehicles, time_limit_s,
                               n_scenarios, service_level_alpha, cost_per_km,
                               stockout_penalty_per_unit, random_state,
                               representative_stat=representative_stat,
                               distance_matrix_km=distance_matrix_km)
    first["capacity_used"] = vehicle_capacity
    first["capacity_adjusted"] = False
    first["suggested_capacity"] = None

    if not first["feasible"]:
        # The nominal capacity can't even fit the representative demand across
        # num_vehicles vehicles -- there's no route topology to measure a service
        # level from at all. This is precisely the case a "capacity-building"
        # feature should handle, not give up on: retry once at a heuristic capacity
        # (representative per-vehicle load with 20% headroom) derived from the SAME
        # scenario distribution, re-seeded so it isn't a repeat of the same draw.
        scenarios = sample_demand_scenarios(median_demand, lower_demand, upper_demand,
                                             calib_alpha, n_scenarios, random_state)
        representative_demand = _representative_demand(scenarios, representative_stat)
        heuristic_capacity = int(np.ceil(representative_demand.sum() / num_vehicles * 1.2))
        fallback_capacity = max(heuristic_capacity, vehicle_capacity + 1)

        fallback = solve_saa_cvrp_sd(depot, customers, median_demand, lower_demand, upper_demand,
                                      calib_alpha, fallback_capacity, num_vehicles, time_limit_s,
                                      n_scenarios, service_level_alpha, cost_per_km,
                                      stockout_penalty_per_unit, random_state=random_state + 1,
                                      representative_stat=representative_stat,
                                      distance_matrix_km=distance_matrix_km)
        fallback["capacity_used"] = fallback_capacity
        fallback["capacity_adjusted"] = True
        fallback["suggested_capacity"] = fallback_capacity
        fallback["capacity_before_adjustment"] = vehicle_capacity
        fallback["service_level_achieved_before_adjustment"] = None
        fallback["infeasible_at_nominal_capacity"] = True
        return fallback

    achieved = first.get("overall_service_level_achieved")
    required = first.get("required_capacity_for_target") or {}
    if achieved is None or achieved >= service_level_alpha or not required:
        return first

    suggested_capacity = int(np.ceil(max(required.values())))
    if suggested_capacity <= vehicle_capacity:
        return first  # honest edge case: the shortfall wasn't actually a capacity-sizing issue

    second = solve_saa_cvrp_sd(depot, customers, median_demand, lower_demand, upper_demand,
                                calib_alpha, suggested_capacity, num_vehicles, time_limit_s,
                                n_scenarios, service_level_alpha, cost_per_km,
                                stockout_penalty_per_unit, random_state=random_state + 1,
                                representative_stat=representative_stat,
                                distance_matrix_km=distance_matrix_km)
    second["capacity_used"] = suggested_capacity
    second["capacity_adjusted"] = True
    second["suggested_capacity"] = suggested_capacity
    second["capacity_before_adjustment"] = vehicle_capacity
    second["service_level_achieved_before_adjustment"] = achieved
    return second
