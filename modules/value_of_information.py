"""Value of the Stochastic Solution (VSS) and Expected Value of Perfect Information
(EVPI) — the standard two-stage stochastic programming yardsticks (Birge & Louveaux,
"Introduction to Stochastic Programming") for whether the ML+OR integration is
actually worth it, versus just eyeballing a chart.

Three plans are compared, all evaluated under the SAME demand scenarios for a fair
comparison:
  - EV  (deterministic, mean-demand): solve routing once on the scenario mean,
    ignoring the distribution entirely — what a naive planner would do.
  - RP  (stochastic/SAA solution): an already-solved SAA plan (see
    `modules/optimization_saa.py`), evaluated the same way.
  - WS  (wait-and-see): solve routing per-scenario with perfect foreknowledge of
    that scenario's realized demand — the best any plan could possibly do.

VSS = EEV - RP  (expected cost of the EV plan under uncertainty, minus the
                 stochastic plan's expected cost) -- should be >= 0.
EVPI = RP - WS  (how much the stochastic plan could still improve with perfect
                 foresight) -- should be >= 0.

Both are reported as-is, including if a numerical fluke on a small demo dataset
makes one come out negative or near zero -- that is itself a valid, reportable
finding (Monte Carlo noise, or the EV/RP plans coinciding), not something to hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.optimization import solve_cvrp_sd
from modules.optimization_saa import evaluate_routes_against_scenarios


def _customer_id_to_idx(customers: pd.DataFrame) -> dict:
    return {cid: i for i, cid in enumerate(customers["customer_id"].to_numpy())}


def evaluate_deterministic_mean_plan(depot: dict, customers: pd.DataFrame, scenarios: np.ndarray,
                                      vehicle_capacity: int, num_vehicles: int, time_limit_s: int = 5,
                                      cost_per_km: float = 1.0, stockout_penalty_per_unit: float = 5.0
                                      ) -> dict:
    """The classic 'EV problem': solve on the scenario mean only, then evaluate that
    fixed plan's expected total cost under the true scenario distribution (this is
    the EEV -- expected result of using the EV solution)."""
    mean_demand = scenarios.mean(axis=0)
    plan = solve_cvrp_sd(depot, customers, mean_demand, vehicle_capacity, num_vehicles, time_limit_s)
    if not plan["feasible"] or not plan["routes"]:
        return {"plan": plan, "evaluation": None, "expected_total_cost": None}

    evaluation = evaluate_routes_against_scenarios(
        plan["routes"], scenarios, _customer_id_to_idx(customers), depot, vehicle_capacity,
        service_level_alpha=0.95, cost_per_km=cost_per_km,
        stockout_penalty_per_unit=stockout_penalty_per_unit)
    expected_total_cost = plan["total_distance_km"] * cost_per_km + evaluation["expected_recourse_cost"]
    return {"plan": plan, "evaluation": evaluation, "expected_total_cost": expected_total_cost}


def evaluate_stochastic_plan(saa_result: dict, scenarios: np.ndarray, customers: pd.DataFrame,
                              depot: dict, vehicle_capacity: int, cost_per_km: float = 1.0,
                              stockout_penalty_per_unit: float = 5.0) -> dict:
    """Evaluates an already-solved SAA plan (`optimization_saa.solve_saa_cvrp_sd`
    output) under the SAME scenario set, for an apples-to-apples RP cost."""
    if not saa_result["feasible"] or not saa_result["routes"]:
        return {"expected_total_cost": None}
    evaluation = evaluate_routes_against_scenarios(
        saa_result["routes"], scenarios, _customer_id_to_idx(customers), depot, vehicle_capacity,
        service_level_alpha=0.95, cost_per_km=cost_per_km,
        stockout_penalty_per_unit=stockout_penalty_per_unit)
    expected_total_cost = saa_result["total_distance_km"] * cost_per_km + evaluation["expected_recourse_cost"]
    return {"evaluation": evaluation, "expected_total_cost": expected_total_cost}


def wait_and_see_cost(depot: dict, customers: pd.DataFrame, scenarios: np.ndarray,
                       vehicle_capacity: int, num_vehicles: int, n_scenarios_ws: int = 20,
                       time_limit_s: int = 1, cost_per_km: float = 1.0, random_state: int = 42) -> dict:
    """Solves CVRP per-scenario with perfect foreknowledge of that scenario's realized
    demand (no recourse needed -- the plan is built exactly for that demand). Re-solving
    OR-Tools per scenario is expensive, so this deliberately uses a SMALLER scenario
    subsample and shorter time limit than the Monte Carlo evaluation elsewhere in this
    module -- a documented speed/accuracy tradeoff, not a hidden shortcut."""
    rng = np.random.default_rng(random_state)
    n = min(n_scenarios_ws, scenarios.shape[0])
    chosen = rng.choice(scenarios.shape[0], size=n, replace=False)

    costs, n_infeasible = [], 0
    for i in chosen:
        plan = solve_cvrp_sd(depot, customers, scenarios[i], vehicle_capacity, num_vehicles, time_limit_s)
        if plan["feasible"]:
            costs.append(plan["total_distance_km"] * cost_per_km)
        else:
            n_infeasible += 1

    return {
        "expected_cost": float(np.mean(costs)) if costs else None,
        "n_used": len(costs), "n_infeasible": n_infeasible, "n_requested": int(n),
    }


def compute_vss_evpi_report(depot: dict, customers: pd.DataFrame, scenarios: np.ndarray,
                             saa_result: dict, vehicle_capacity: int, num_vehicles: int,
                             time_limit_s: int = 5, cost_per_km: float = 1.0,
                             stockout_penalty_per_unit: float = 5.0, n_scenarios_ws: int = 20,
                             ws_time_limit_s: int = 1, random_state: int = 42) -> dict:
    """Full VSS/EVPI report: the single most defensible "does the ML+OR integration
    matter" evidence this project can produce, since it's a direct expected-cost
    comparison rather than a single-scenario anecdote."""
    ev = evaluate_deterministic_mean_plan(depot, customers, scenarios, vehicle_capacity,
                                           num_vehicles, time_limit_s, cost_per_km,
                                           stockout_penalty_per_unit)
    rp = evaluate_stochastic_plan(saa_result, scenarios, customers, depot, vehicle_capacity,
                                   cost_per_km, stockout_penalty_per_unit)
    ws = wait_and_see_cost(depot, customers, scenarios, vehicle_capacity, num_vehicles,
                            n_scenarios_ws, ws_time_limit_s, cost_per_km, random_state)

    eev_cost, rp_cost, ws_cost = ev["expected_total_cost"], rp["expected_total_cost"], ws["expected_cost"]
    vss = (eev_cost - rp_cost) if (eev_cost is not None and rp_cost is not None) else None
    evpi = (rp_cost - ws_cost) if (rp_cost is not None and ws_cost is not None) else None

    return {
        "EEV_cost": eev_cost, "RP_cost": rp_cost, "WS_cost": ws_cost,
        "VSS": vss, "EVPI": evpi,
        "deterministic_plan": ev, "stochastic_evaluation": rp, "wait_and_see": ws,
    }
