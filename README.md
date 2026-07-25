# Resilient Supply Chain Cockpit

An integrated Machine Learning + Operations Research system for mitigating supply chain
disruptions: probabilistic demand forecasting hedges a stochastic vehicle routing plan,
with explainability (XAI) surfaced throughout an interactive dashboard.

## Architecture

Four layers, wired end-to-end:

1. **Data Ingestion** (`data/generate_data.py`) — synthetic daily data for a Bogotá-area
   logistics network: multi-customer demand, a Geopolitical Volatility Index (GVI),
   climate and macroeconomic indices, per-route lead times, and depot/customer
   coordinates. Cached to CSV on first run.
2. **Machine Learning** (`modules/forecasting.py`) — two independent probabilistic
   forecasters behind a common interface:
   - `QuantileXGBForecaster` / `XGBQuantileEnsemble` — XGBoost with the native
     `reg:quantileerror` objective, fitting an arbitrary target quantile (default
     q=0.90) for upper-bound demand estimates.
   - `LSTMForecaster` — a PyTorch LSTM trained with pinball loss, emitting the
     q0.10 / q0.50 / q0.90 quantiles in one forward pass.
   - `compute_stockout_risk` — approximates P(demand > capacity) from a normal fit to
     each forecast's (q10, q50, q90) band.
3. **Optimization** (`modules/optimization.py`) — Capacitated Vehicle Routing with
   Stochastic Demand (CVRP-SD), solved with OR-Tools (`PATH_CHEAPEST_ARC` first
   solution + guided local search). Stochastic demand is handled via a chance-constrained
   approximation: the capacity dimension is built from the forecaster's upper-quantile
   demand rather than the mean, hedging routes against demand up to that quantile.
4. **Web UI + XAI** (`app.py`, `modules/xai_engine.py`) — a Streamlit dashboard with
   demand-trajectory charts, stockout-risk KPIs, a Folium routing map, and SHAP/LIME
   explanations of the forecasting model.

## Directory Structure

```
doctoral_supply_chain/
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── generate_data.py       # synthetic data generator + loader (writes CSVs here)
├── modules/
│   ├── __init__.py
│   ├── forecasting.py         # XGBoost quantile + LSTM quantile forecasters
│   ├── optimization.py        # CVRP-SD solver (OR-Tools)
│   └── xai_engine.py          # SHAP + LIME explainability
└── app.py                     # Streamlit dashboard
```

## Setup

Requires Python 3.11+ (XGBoost's `reg:quantileerror` objective needs XGBoost ≥ 2.0).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Note: `torch` and `xgboost` wheels are large (~100–200 MB each); on a slow connection
> add `--timeout 180 --retries 8` to the `pip install` command.

## Usage

Generate the synthetic dataset (optional — `app.py` does this automatically on first
run):

```bash
python data/generate_data.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

### Data source

- **Synthetic demo** — the built-in Bogotá-area network. **Simulated customers** (5–60)
  and **Simulated history (days)** (90–730) control the size of the generated dataset;
  smaller values mean faster training/routing/SHAP while developing.
- **Upload my own dataset** — bring your own logistics data and run the full pipeline
  (forecast → risk → routing → XAI) against it:
  - `customers.csv` (required): `customer_id, name, lat, lon, base_demand`
  - `demand.csv` (required): `date, customer_id, demand` — at least 15 daily rows per
    customer (the forecasters need that much history for lag/rolling features)
  - `exogenous.csv` (optional): `date, gvi, climate_index, macro_index` — omit it and
    flat neutral values are used instead
  - Depot latitude/longitude default to the centroid of the uploaded customers and can
    be overridden in the sidebar.
  - CSV templates are downloadable from the "Required CSV schema" sidebar expander.
    Invalid uploads (missing columns, unparseable dates, unknown customer IDs, too
    little history) show specific error messages instead of crashing.
  - A "Dataset summary" panel at the top of the main page shows customer count, row
    count, and date range for whichever dataset is active.

### Performance mode

- **Fast** (default) — smaller SHAP sample, shorter CVRP-SD solve time limit, a
  30-customer cap on routing, and lighter XGBoost/LSTM models. Recommended on free-tier
  cloud hosting (e.g. Streamlit Community Cloud) to stay under CPU throttling limits.
- **Full** — production-quality settings: full SHAP sample, longer solve time, no
  customer cap, full model size.

### Dashboard controls

- **Geopolitical Volatility Index** — overrides the last 30 days of GVI to simulate a
  disruption shock; re-trains the forecast and propagates into routing.
- **ML Quantile Target** — the upper-bound demand quantile (0.50–0.99) used both to size
  the uncertainty band shown in the forecast chart and as the effective demand fed into
  the CVRP-SD capacity constraint.
- **Vehicle Capacity** — per-vehicle capacity used by the routing solver.

### Panels

- **Demand Forecast** — actual vs. quantile-forecast demand trajectory, with an optional
  LSTM comparison model (trained on demand, cached after first click), and a CSV export
  of the plotted forecast.
- **Stockout Risk** — naive mean-demand inventory sizing vs. ML-informed
  (quantile-sized) inventory, with the relative risk reduction computed live from the
  current scenario (not a fixed benchmark number), and a CSV export of the per-customer
  risk table.
- **Vehicle Routing** — the CVRP-SD solution rendered on a Folium map, color-coded per
  vehicle, sized against the current quantile-target demand forecast, with a CSV export
  of the per-stop routing plan. In Fast mode, routing is capped to a subsample of
  customers (see Performance mode above).
- **Explainability** — SHAP global feature importance and per-customer SHAP/LIME
  explanations for the active XGBoost quantile model.

### Performance notes

CVRP-SD solving (OR-Tools) and SHAP/LIME explanation are cached (`@st.cache_data`) keyed
on their actual inputs, and model training is cached (`@st.cache_resource`) keyed on the
active dataset plus the scenario parameters that affect it — so unrelated widget changes
(e.g. moving the vehicle capacity slider while on the Forecast tab) no longer re-solve
routing or recompute explanations on every rerun.

## Design notes

- **OR-Tools only.** No Gurobi dependency/license is required; the routing solver runs
  entirely on the open-source OR-Tools CP-SAT/routing library.
- **`reg:quantileerror`, not `reg:absoluteerror`.** MAE loss only targets the median
  (q=0.5) and cannot produce an upper-bound estimate; targeting q=0.90 requires
  XGBoost's native quantile objective with `quantile_alpha`.
- **Independent forecasters.** XGBoost and LSTM are not stacked — they share an
  interface so the dashboard can compare them, not so one feeds the other.
