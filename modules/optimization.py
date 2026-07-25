"""CVRP with stochastic demand, solved via OR-Tools.

Stochastic demand is handled with a chance-constrained approximation: the vehicle
capacity dimension is built from the forecaster's upper-quantile (e.g. q=0.90) demand
estimate rather than the mean, so routes are hedged against demand realizations up to
that quantile — a tractable stand-in for full two-stage recourse optimization.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

EARTH_RADIUS_KM = 6371.0


def _haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat = np.radians(lats)[:, None]
    lon = np.radians(lons)[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.clip(np.sqrt(a), -1, 1))


def suggest_fleet_size(demand_forecast: np.ndarray, vehicle_capacity: int, buffer: float = 1.3) -> int:
    """Rough fleet-size heuristic: total demand / capacity, with slack for routing inefficiency."""
    return max(1, math.ceil(np.sum(demand_forecast) / vehicle_capacity * buffer))


def solve_cvrp_sd(depot: dict, customers: pd.DataFrame, demand_forecast: np.ndarray,
                   vehicle_capacity: int, num_vehicles: int, time_limit_s: int = 5) -> dict:
    locations = pd.concat([
        pd.DataFrame([{"customer_id": depot["customer_id"], "lat": depot["lat"], "lon": depot["lon"]}]),
        customers[["customer_id", "lat", "lon"]],
    ], ignore_index=True)

    dist_km = _haversine_matrix(locations["lat"].to_numpy(), locations["lon"].to_numpy())
    dist_m = np.round(dist_km * 1000).astype(int)
    demands = np.concatenate([[0], np.round(np.asarray(demand_forecast)).astype(int)])

    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(dist_m[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        return int(demands[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [vehicle_capacity] * num_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"routes": [], "total_distance_km": 0.0, "total_demand_served": 0,
                "num_vehicles_used": 0, "feasible": False}

    routes, total_distance_m = [], 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        nodes, route_distance, route_load = [], 0, 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            nodes.append(node)
            route_load += demands[node]
            prev_index, index = index, solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        nodes.append(manager.IndexToNode(index))
        total_distance_m += route_distance
        if len(nodes) > 2:
            routes.append({
                "vehicle_id": vehicle_id,
                "stops": [int(locations.iloc[n]["customer_id"]) for n in nodes],
                "coords": [(float(locations.iloc[n]["lat"]), float(locations.iloc[n]["lon"])) for n in nodes],
                "load": int(route_load),
                "distance_km": round(route_distance / 1000, 2),
            })

    return {
        "routes": routes,
        "total_distance_km": round(total_distance_m / 1000, 2),
        "total_demand_served": int(demands.sum()),
        "num_vehicles_used": len(routes),
        "feasible": True,
    }
