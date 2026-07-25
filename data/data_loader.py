"""Bring-your-own-dataset support: validation, gap-filling, and templates for the
customers.csv / demand.csv / (optional) exogenous.csv upload path in app.py.

Mirrors the schema produced by `generate_data.py` so uploaded data flows through the
existing forecasting/routing/XAI pipeline unchanged. `lead_times.csv` is intentionally
not part of this contract — the dashboard doesn't read it anywhere today.
"""
from __future__ import annotations

import io

import pandas as pd

REQUIRED_CUSTOMER_COLS = ["customer_id", "name", "lat", "lon", "base_demand"]
REQUIRED_DEMAND_COLS = ["date", "customer_id", "demand"]
EXOGENOUS_COLS = ["gvi", "climate_index", "macro_index"]
MIN_HISTORY_DAYS = 15  # forecasting.py lag_14 / LSTMForecaster lookback=14 need >= this many rows/customer


def _missing_cols(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def validate_uploaded_data(customers_df: pd.DataFrame, demand_df: pd.DataFrame,
                            exogenous_df: pd.DataFrame | None = None) -> list[str]:
    """Returns a list of human-readable error messages; empty list means the data is usable."""
    errors: list[str] = []

    missing = _missing_cols(customers_df, REQUIRED_CUSTOMER_COLS)
    if missing:
        errors.append(f"customers.csv is missing required column(s): {', '.join(missing)}")
    missing = _missing_cols(demand_df, REQUIRED_DEMAND_COLS)
    if missing:
        errors.append(f"demand.csv is missing required column(s): {', '.join(missing)}")
    if errors:
        # Can't safely check anything else until the required columns actually exist.
        return errors

    if pd.to_numeric(customers_df["customer_id"], errors="coerce").isna().any():
        errors.append("customers.csv: 'customer_id' must be numeric (the model uses it "
                       "directly as a numeric feature).")
    if pd.to_numeric(demand_df["customer_id"], errors="coerce").isna().any():
        errors.append("demand.csv: 'customer_id' must be numeric.")
    if customers_df["customer_id"].duplicated().any():
        errors.append("customers.csv: duplicate 'customer_id' values found.")

    for col in ["lat", "lon", "base_demand"]:
        if pd.to_numeric(customers_df[col], errors="coerce").isna().any():
            errors.append(f"customers.csv: '{col}' has non-numeric or missing values.")

    demand_dates = pd.to_datetime(demand_df["date"], errors="coerce")
    if demand_dates.isna().any():
        errors.append("demand.csv: 'date' has values that can't be parsed as dates.")
    if pd.to_numeric(demand_df["demand"], errors="coerce").isna().any():
        errors.append("demand.csv: 'demand' has non-numeric or missing values.")

    if not errors:
        unknown_ids = set(demand_df["customer_id"]) - set(customers_df["customer_id"])
        if unknown_ids:
            sample = ", ".join(str(i) for i in list(unknown_ids)[:5])
            errors.append(f"demand.csv references customer_id(s) not present in customers.csv: "
                           f"{sample}{'...' if len(unknown_ids) > 5 else ''}")

        history_len = demand_df.groupby("customer_id")["date"].nunique()
        thin = history_len[history_len < MIN_HISTORY_DAYS]
        if not thin.empty:
            errors.append(f"demand.csv: {len(thin)} customer(s) have fewer than "
                           f"{MIN_HISTORY_DAYS} days of history, which isn't enough to build "
                           f"lag/rolling features (need >= {MIN_HISTORY_DAYS} daily rows per customer).")

    if exogenous_df is not None:
        if "date" not in exogenous_df.columns:
            errors.append("exogenous.csv: missing required column 'date'.")
        elif pd.to_datetime(exogenous_df["date"], errors="coerce").isna().any():
            errors.append("exogenous.csv: 'date' has values that can't be parsed as dates.")
        for col in EXOGENOUS_COLS:
            if col in exogenous_df.columns and pd.to_numeric(exogenous_df[col], errors="coerce").isna().any():
                errors.append(f"exogenous.csv: '{col}' has non-numeric values.")

    return errors


def fill_missing_exogenous_columns(exogenous_df: pd.DataFrame | None,
                                    dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """Ensures gvi/climate_index/macro_index exist, filling neutral placeholder values for
    whatever wasn't supplied so the existing forecasting pipeline still runs. Returns the
    completed frame plus a list of warning strings for anything that was filled in."""
    warnings: list[str] = []
    neutral_defaults = {"gvi": 50.0, "climate_index": 50.0, "macro_index": 100.0}

    if exogenous_df is None:
        warnings.append("No exogenous.csv provided — using flat neutral values for GVI, "
                         "climate index, and macro index (the GVI slider will still work, "
                         "but starts from a constant baseline instead of real history).")
        return pd.DataFrame({"date": dates, **{k: v for k, v in neutral_defaults.items()}}), warnings

    out = exogenous_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col, default in neutral_defaults.items():
        if col not in out.columns:
            warnings.append(f"exogenous.csv had no '{col}' column — filled with a flat value of {default}.")
            out[col] = default

    full = pd.DataFrame({"date": dates})
    out = full.merge(out[["date", *EXOGENOUS_COLS]], on="date", how="left")
    if out[EXOGENOUS_COLS].isna().any().any():
        warnings.append("exogenous.csv doesn't cover every date in demand.csv — gaps were "
                         "forward/back-filled with neutral values.")
        for col, default in neutral_defaults.items():
            out[col] = out[col].ffill().bfill().fillna(default)
    return out, warnings


def centroid_depot(customers_df: pd.DataFrame) -> dict:
    return {
        "customer_id": 0,
        "name": "Depot (auto-centered on uploaded customers)",
        "lat": float(customers_df["lat"].mean()),
        "lon": float(customers_df["lon"].mean()),
    }


def build_uploaded_dataset(customers_df: pd.DataFrame, demand_df: pd.DataFrame,
                            exogenous_df: pd.DataFrame | None, depot: dict) -> tuple[dict, list[str]]:
    """Assembles the same dict shape `generate_data.load_or_generate_data` returns, from
    validated uploaded frames. Caller must run `validate_uploaded_data` first."""
    customers = customers_df.copy()
    customers["customer_id"] = pd.to_numeric(customers["customer_id"]).astype(int)

    demand = demand_df.copy()
    demand["customer_id"] = pd.to_numeric(demand["customer_id"]).astype(int)
    demand["date"] = pd.to_datetime(demand["date"])
    demand["demand"] = pd.to_numeric(demand["demand"])

    dates = pd.DatetimeIndex(sorted(demand["date"].unique()))
    exogenous, warnings = fill_missing_exogenous_columns(exogenous_df, dates)

    return {
        "demand": demand,
        "exogenous": exogenous,
        "customers": customers,
        "lead_times": None,
        "depot": depot,
    }, warnings


def template_csv_bytes() -> dict[str, bytes]:
    """Header + one example row per file, for the sidebar template download buttons."""
    customers = pd.DataFrame([
        {"customer_id": 1, "name": "Customer 1", "lat": 4.7110, "lon": -74.0721, "base_demand": 50},
    ])
    demand = pd.DataFrame([
        {"date": "2024-01-01", "customer_id": 1, "demand": 48.5},
    ])
    exogenous = pd.DataFrame([
        {"date": "2024-01-01", "gvi": 45.0, "climate_index": 52.0, "macro_index": 100.0},
    ])
    out = {}
    for name, df in [("customers.csv", customers), ("demand.csv", demand), ("exogenous.csv", exogenous)]:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        out[name] = buf.getvalue().encode("utf-8")
    return out
