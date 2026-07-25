import pandas as pd
import pytest

from data.data_loader import (build_uploaded_dataset, centroid_depot, fill_missing_exogenous_columns,
                               template_csv_bytes, validate_uploaded_data)


def _valid_customers():
    return pd.DataFrame({
        "customer_id": [1, 2], "name": ["A", "B"], "lat": [4.71, 4.72],
        "lon": [-74.07, -74.08], "base_demand": [50, 60],
    })


def _valid_demand(n_days=20):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for cid in (1, 2):
        for d in dates:
            rows.append({"date": d, "customer_id": cid, "demand": 10.0})
    return pd.DataFrame(rows)


def test_validate_uploaded_data_accepts_well_formed_input():
    errors = validate_uploaded_data(_valid_customers(), _valid_demand())
    assert errors == []


def test_validate_rejects_missing_customer_columns():
    bad_customers = _valid_customers().drop(columns=["lat"])
    errors = validate_uploaded_data(bad_customers, _valid_demand())
    assert any("lat" in e for e in errors)


def test_validate_rejects_missing_demand_columns():
    bad_demand = _valid_demand().drop(columns=["demand"])
    errors = validate_uploaded_data(_valid_customers(), bad_demand)
    assert any("demand" in e for e in errors)


def test_validate_rejects_non_numeric_customer_id():
    bad_customers = _valid_customers().copy()
    bad_customers["customer_id"] = ["one", "two"]
    errors = validate_uploaded_data(bad_customers, _valid_demand())
    assert any("customer_id" in e and "numeric" in e for e in errors)


def test_validate_rejects_duplicate_customer_ids():
    bad_customers = _valid_customers().copy()
    bad_customers["customer_id"] = [1, 1]
    errors = validate_uploaded_data(bad_customers, _valid_demand())
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_rejects_unparseable_dates():
    bad_demand = _valid_demand().copy()
    bad_demand["date"] = bad_demand["date"].astype(object)
    bad_demand.loc[0, "date"] = "not-a-date"
    errors = validate_uploaded_data(_valid_customers(), bad_demand)
    assert any("date" in e for e in errors)


def test_validate_rejects_demand_referencing_unknown_customer():
    bad_demand = _valid_demand().copy()
    bad_demand.loc[0, "customer_id"] = 999
    errors = validate_uploaded_data(_valid_customers(), bad_demand)
    assert any("not present in customers.csv" in e for e in errors)


def test_validate_rejects_too_little_history():
    thin_demand = _valid_demand(n_days=5)
    errors = validate_uploaded_data(_valid_customers(), thin_demand)
    assert any("fewer than" in e for e in errors)


def test_validate_accepts_missing_optional_exogenous():
    errors = validate_uploaded_data(_valid_customers(), _valid_demand(), exogenous_df=None)
    assert errors == []


def test_validate_rejects_bad_exogenous_columns():
    bad_exo = pd.DataFrame({"date": ["2024-01-01"], "gvi": ["not-a-number"]})
    errors = validate_uploaded_data(_valid_customers(), _valid_demand(), exogenous_df=bad_exo)
    assert any("gvi" in e for e in errors)


def test_fill_missing_exogenous_columns_uses_neutral_defaults():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    filled, warnings = fill_missing_exogenous_columns(None, dates)
    assert len(warnings) == 1
    assert (filled["gvi"] == 50.0).all()
    assert (filled["macro_index"] == 100.0).all()


def test_centroid_depot_is_mean_of_customers():
    depot = centroid_depot(_valid_customers())
    assert depot["lat"] == pytest.approx(4.715)
    assert depot["lon"] == pytest.approx(-74.075)


def test_build_uploaded_dataset_matches_generator_schema():
    depot = {"customer_id": 0, "name": "Depot", "lat": 4.71, "lon": -74.07}
    data, warnings = build_uploaded_dataset(_valid_customers(), _valid_demand(), None, depot)
    assert set(data.keys()) == {"demand", "exogenous", "customers", "lead_times", "depot"}
    assert len(warnings) == 1  # "no exogenous.csv provided" warning


def test_template_csv_bytes_are_parseable():
    templates = template_csv_bytes()
    assert set(templates.keys()) == {"customers.csv", "demand.csv", "exogenous.csv"}
    for content in templates.values():
        assert len(content) > 0
