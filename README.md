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

### Dashboard controls

- **Geopolitical Volatility Index** — overrides the last 30 days of GVI to simulate a
  disruption shock; re-trains the forecast and propagates into routing.
- **ML Quantile Target** — the upper-bound demand quantile (0.50–0.99) used both to size
  the uncertainty band shown in the forecast chart and as the effective demand fed into
  the CVRP-SD capacity constraint.
- **Vehicle Capacity** — per-vehicle capacity used by the routing solver.

### Panels

- **Demand Forecast** — actual vs. quantile-forecast demand trajectory, with an optional
  LSTM comparison model (trained on demand, cached after first click).
- **Stockout Risk** — naive mean-demand inventory sizing vs. ML-informed
  (quantile-sized) inventory, with the relative risk reduction computed live from the
  current scenario (not a fixed benchmark number).
- **Vehicle Routing** — the CVRP-SD solution rendered on a Folium map, color-coded per
  vehicle, sized against the current quantile-target demand forecast.
- **Explainability** — SHAP global feature importance and per-customer SHAP/LIME
  explanations for the active XGBoost quantile model.

## Design notes

- **OR-Tools only.** No Gurobi dependency/license is required; the routing solver runs
  entirely on the open-source OR-Tools CP-SAT/routing library.
- **`reg:quantileerror`, not `reg:absoluteerror`.** MAE loss only targets the median
  (q=0.5) and cannot produce an upper-bound estimate; targeting q=0.90 requires
  XGBoost's native quantile objective with `quantile_alpha`.
- **Independent forecasters.** XGBoost and LSTM are not stacked — they share an
  interface so the dashboard can compare them, not so one feeds the other.
