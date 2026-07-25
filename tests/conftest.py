"""Shared pytest fixtures. Deliberately builds synthetic data via the LOW-LEVEL
generator functions (generate_customer_locations / generate_exogenous_series /
generate_demand_data) rather than `load_or_generate_data`, which caches to CSV on
disk under data/ — tests should never write into the project's data directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from data.generate_data import (generate_customer_locations, generate_demand_data,
                                 generate_exogenous_series)
from modules.forecasting import build_feature_frame


@pytest.fixture(scope="session")
def small_synthetic_data():
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    customers = generate_customer_locations(n_customers=6, seed=1)
    exogenous = generate_exogenous_series(dates, seed=1)
    demand = generate_demand_data(dates, customers, exogenous, seed=1)
    depot = {"customer_id": 0, "name": "Test depot", "lat": 4.7110, "lon": -74.0721}
    return {"demand": demand, "exogenous": exogenous, "customers": customers, "depot": depot}


@pytest.fixture(scope="session")
def small_features(small_synthetic_data):
    return build_feature_frame(small_synthetic_data["demand"], small_synthetic_data["exogenous"])
