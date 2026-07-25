"""Synthetic data bootstrap: Bogotá-area logistics network + demand/risk drivers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEPOT = {"customer_id": 0, "name": "Depot - Bogotá DC", "lat": 4.7110, "lon": -74.0721}
N_DAYS = 730
SEED = 42


def _date_index(n_days: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n_days, freq="D")


def generate_exogenous_series(dates: pd.DatetimeIndex, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(len(dates))
    gvi = 40 + 15 * np.sin(t / 60) + rng.normal(0, 6, len(t))
    climate_index = 50 + 20 * np.sin(2 * np.pi * t / 365 + 0.5) + rng.normal(0, 5, len(t))
    macro_index = 100 + 0.03 * t + 8 * np.sin(t / 120) + rng.normal(0, 3, len(t))
    return pd.DataFrame({
        "date": dates,
        "gvi": gvi.clip(0, 100),
        "climate_index": climate_index.clip(0, 100),
        "macro_index": macro_index,
    })


def generate_customer_locations(n_customers: int = 24, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    radius_deg = 0.28  # ~30km around Bogotá
    angles = rng.uniform(0, 2 * np.pi, n_customers)
    radii = radius_deg * np.sqrt(rng.uniform(0.05, 1.0, n_customers))
    lat = DEPOT["lat"] + radii * np.cos(angles)
    lon = DEPOT["lon"] + radii * np.sin(angles) * 1.1  # slight lon stretch at this latitude
    return pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "name": [f"Customer {i}" for i in range(1, n_customers + 1)],
        "lat": lat,
        "lon": lon,
        "base_demand": rng.integers(20, 90, n_customers),
        "gvi_sensitivity": rng.uniform(-0.3, 0.6, n_customers),
        "climate_sensitivity": rng.uniform(-0.2, 0.4, n_customers),
    })


def generate_demand_data(dates: pd.DatetimeIndex, customers: pd.DataFrame,
                          exogenous: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    t = np.arange(len(dates))
    dow = dates.dayofweek.values
    rows = []
    for _, cust in customers.iterrows():
        trend = 1 + 0.0004 * t
        weekly = 1 + 0.15 * np.isin(dow, [4, 5]).astype(float)
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 365)
        gvi_effect = 1 + cust["gvi_sensitivity"] * (exogenous["gvi"].values - 40) / 100
        climate_effect = 1 + cust["climate_sensitivity"] * (exogenous["climate_index"].values - 50) / 100
        noise = rng.normal(1, 0.18, len(t))
        demand = (cust["base_demand"] * trend * weekly * seasonal * gvi_effect
                  * climate_effect * noise).clip(min=0)
        rows.append(pd.DataFrame({
            "date": dates,
            "customer_id": cust["customer_id"],
            "demand": np.round(demand, 1),
        }))
    return pd.concat(rows, ignore_index=True)


def generate_lead_times(customers: pd.DataFrame, exogenous: pd.DataFrame,
                         seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    rows = []
    for _, cust in customers.iterrows():
        base = rng.uniform(1.5, 4.5)
        gvi_penalty = 0.02 * exogenous["gvi"].values
        lead_time = base + gvi_penalty + rng.normal(0, 0.3, len(exogenous))
        rows.append(pd.DataFrame({
            "date": exogenous["date"],
            "customer_id": cust["customer_id"],
            "lead_time_days": np.round(lead_time.clip(min=0.5), 2),
        }))
    return pd.concat(rows, ignore_index=True)


def load_or_generate_data(n_customers: int = 24, n_days: int = N_DAYS, seed: int = SEED,
                           force: bool = False) -> dict:
    tag = f"{n_customers}c_{n_days}d_{seed}s"
    paths = {
        "demand": os.path.join(DATA_DIR, f"demand_{tag}.csv"),
        "exogenous": os.path.join(DATA_DIR, f"exogenous_{tag}.csv"),
        "customers": os.path.join(DATA_DIR, f"customers_{tag}.csv"),
        "lead_times": os.path.join(DATA_DIR, f"lead_times_{tag}.csv"),
    }
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {
            "demand": pd.read_csv(paths["demand"], parse_dates=["date"]),
            "exogenous": pd.read_csv(paths["exogenous"], parse_dates=["date"]),
            "customers": pd.read_csv(paths["customers"]),
            "lead_times": pd.read_csv(paths["lead_times"], parse_dates=["date"]),
            "depot": DEPOT,
        }

    dates = _date_index(n_days)
    exogenous = generate_exogenous_series(dates, seed)
    customers = generate_customer_locations(n_customers, seed)
    demand = generate_demand_data(dates, customers, exogenous, seed)
    lead_times = generate_lead_times(customers, exogenous, seed)

    exogenous.to_csv(paths["exogenous"], index=False)
    customers.to_csv(paths["customers"], index=False)
    demand.to_csv(paths["demand"], index=False)
    lead_times.to_csv(paths["lead_times"], index=False)

    return {"demand": demand, "exogenous": exogenous, "customers": customers,
            "lead_times": lead_times, "depot": DEPOT}


if __name__ == "__main__":
    data = load_or_generate_data(force=True)
    for key, val in data.items():
        if isinstance(val, pd.DataFrame):
            print(f"{key}: {val.shape}")
        else:
            print(f"{key}: {val}")
