import numpy as np
import pandas as pd
import pytest

from modules.optimization import solve_cvrp_sd, suggest_fleet_size
from modules.optimization_saa import (evaluate_routes_against_scenarios, sample_demand_scenarios,
                                       solve_saa_cvrp_sd, solve_saa_cvrp_sd_target_capacity)
from modules.value_of_information import compute_vss_evpi_report


@pytest.fixture
def tiny_instance():
    depot = {"customer_id": 0, "name": "Depot", "lat": 4.71, "lon": -74.07}
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
        "lat": [4.72, 4.70, 4.73, 4.69],
        "lon": [-74.06, -74.08, -74.05, -74.09],
    })
    return depot, customers


def test_solve_cvrp_sd_returns_feasible_routes_respecting_capacity(tiny_instance):
    depot, customers = tiny_instance
    demand = np.array([50, 60, 40, 30])
    capacity = 100
    num_vehicles = suggest_fleet_size(demand, capacity)
    result = solve_cvrp_sd(depot, customers, demand, capacity, num_vehicles, time_limit_s=2)

    assert result["feasible"]
    for route in result["routes"]:
        assert route["load"] <= capacity
    assert result["total_demand_served"] == int(demand.sum())


def test_solve_cvrp_sd_infeasible_returns_clean_result(tiny_instance):
    depot, customers = tiny_instance
    # Impossibly small capacity relative to demand and a single vehicle -> infeasible.
    demand = np.array([500, 500, 500, 500])
    result = solve_cvrp_sd(depot, customers, demand, vehicle_capacity=10, num_vehicles=1, time_limit_s=1)
    assert result["feasible"] is False
    assert result["routes"] == []


def test_solve_cvrp_sd_accepts_custom_distance_matrix(tiny_instance):
    depot, customers = tiny_instance
    demand = np.array([10, 10, 10, 10])
    n = len(customers) + 1
    flat_matrix = np.ones((n, n)) * 5.0
    np.fill_diagonal(flat_matrix, 0.0)
    result = solve_cvrp_sd(depot, customers, demand, vehicle_capacity=100, num_vehicles=1,
                            time_limit_s=2, distance_matrix_km=flat_matrix)
    assert result["feasible"]


def test_solve_cvrp_sd_rejects_wrong_shaped_distance_matrix(tiny_instance):
    depot, customers = tiny_instance
    demand = np.array([10, 10, 10, 10])
    with pytest.raises(ValueError):
        solve_cvrp_sd(depot, customers, demand, vehicle_capacity=100, num_vehicles=1,
                       time_limit_s=1, distance_matrix_km=np.ones((2, 2)))


def test_saa_service_level_achieved_is_in_unit_interval(tiny_instance):
    depot, customers = tiny_instance
    median = np.array([40.0, 45.0, 35.0, 30.0])
    lower = median - 10
    upper = median + 10
    result = solve_saa_cvrp_sd(depot, customers, median, lower, upper, calib_alpha=0.1,
                                vehicle_capacity=100, num_vehicles=2, time_limit_s=2,
                                n_scenarios=100, service_level_alpha=0.9)
    assert result["feasible"]
    assert 0.0 <= result["overall_service_level_achieved"] <= 1.0
    assert 0.0 <= result["mean_route_service_level"] <= 1.0
    assert result["expected_recourse_cost"] >= 0.0


def test_saa_target_capacity_wrapper_reports_capacity_used(tiny_instance):
    depot, customers = tiny_instance
    median = np.array([40.0, 45.0, 35.0, 30.0])
    lower, upper = median - 15, median + 15
    result = solve_saa_cvrp_sd_target_capacity(depot, customers, median, lower, upper, calib_alpha=0.1,
                                                vehicle_capacity=80, num_vehicles=2, time_limit_s=2,
                                                n_scenarios=100, service_level_alpha=0.95)
    assert result["feasible"]
    assert result["capacity_used"] >= 80
    assert isinstance(result["capacity_adjusted"], bool)


def test_sample_demand_scenarios_shape_and_nonnegative():
    median = np.array([50.0, 60.0])
    lower, upper = median - 10, median + 10
    scenarios = sample_demand_scenarios(median, lower, upper, alpha=0.1, n_scenarios=50, random_state=1)
    assert scenarios.shape == (50, 2)
    assert np.all(scenarios >= 0)


def test_vss_evpi_computed_without_error(tiny_instance):
    depot, customers = tiny_instance
    median = np.array([40.0, 45.0, 35.0, 30.0])
    lower, upper = median - 10, median + 10
    scenarios = sample_demand_scenarios(median, lower, upper, alpha=0.1, n_scenarios=80, random_state=2)
    saa_result = solve_saa_cvrp_sd(depot, customers, median, lower, upper, calib_alpha=0.1,
                                   vehicle_capacity=100, num_vehicles=2, time_limit_s=2,
                                   n_scenarios=80, service_level_alpha=0.9)
    report = compute_vss_evpi_report(depot, customers, scenarios, saa_result, vehicle_capacity=100,
                                      num_vehicles=2, time_limit_s=2, n_scenarios_ws=10, ws_time_limit_s=1)
    assert "VSS" in report and "EVPI" in report
    if report["VSS"] is not None:
        assert np.isfinite(report["VSS"])
    if report["EVPI"] is not None:
        assert np.isfinite(report["EVPI"])
