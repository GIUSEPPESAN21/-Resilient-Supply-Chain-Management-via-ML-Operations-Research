# Resilient Supply Chain Cockpit

An integrated Machine Learning + Operations Research system for mitigating supply chain
disruptions: probabilistic demand forecasting hedges a stochastic vehicle routing plan,
with explainability (XAI) and calibration/significance diagnostics surfaced throughout an
interactive dashboard.

## Architecture

Four layers, wired end-to-end:

1. **Data Ingestion** (`data/generate_data.py`) — synthetic daily data for a Bogotá-area
   logistics network: multi-customer demand, a Geopolitical Volatility Index (GVI),
   climate and macroeconomic indices, per-route lead times, and depot/customer
   coordinates. Cached to CSV on first run.
2. **Machine Learning** (`modules/forecasting.py`, `modules/metrics.py`,
   `modules/backtesting.py`, `modules/significance.py`) — probabilistic forecasters
   behind a common interface, plus calibration and significance diagnostics:
   - `QuantileXGBForecaster` / `XGBQuantileEnsemble` — XGBoost with the native
     `reg:quantileerror` objective, fitting independent quantiles (default 0.1/0.5/0.9).
     `enforce_monotonic_quantiles()` applies the Chernozhukov, Fernandez-Val & Galichon
     (2010) rearrangement fix so q10 <= q50 <= q90 holds row-by-row.
   - `ConformalizedQuantileForecaster` — Conformalized Quantile Regression (Romano,
     Patterson & Candes 2019): a time-respecting proper-train/calibration split gives
     `.predict_interval(features, alpha)` a marginal coverage guarantee, on top of the
     existing `.predict_quantiles()` interface. Optional online-adaptive mode (Gibbs &
     Candes 2021) tracks a running miscoverage rate for non-stationary demand.
   - `LSTMForecaster` — a PyTorch LSTM trained with joint pinball loss, emitting the
     q0.10 / q0.50 / q0.90 quantiles in one forward pass.
   - `compute_stockout_risk` — approximates P(demand > capacity) from a normal fit to
     each forecast's (q10, q50, q90) band (still a parametric approximation, unchanged).
   - `modules/backtesting.py` — rolling-origin (walk-forward) cross-validation; reports
     pinball loss / CRPS / empirical coverage per fold, aggregated as mean +/- std.
   - `modules/significance.py` — Diebold-Mariano (2 forecasters) or Friedman + Nemenyi
     (Demsar 2006, >2 forecasters) significance testing via `compare_forecasters()`.
   - `modules/mlflow_tracking.py` — optional MLflow logging of backtest runs (params,
     per-fold metrics, calibration plot); returns `None` gracefully if MLflow isn't
     installed, so nothing else depends on it.
3. **Optimization** (`modules/optimization.py`, `modules/optimization_saa.py`,
   `modules/value_of_information.py`, `modules/routing_backend.py`):
   - `solve_cvrp_sd` (unchanged, still the default) — deterministic CVRP-SD via
     OR-Tools, capacity sized from the forecaster's upper-quantile demand.
   - `solve_saa_cvrp_sd` / `solve_saa_cvrp_sd_target_capacity` — Sample Average
     Approximation: routes on a robust (0.75-quantile) representative demand drawn
     from Monte Carlo scenarios (fed by the calibrated CQR interval), Monte
     Carlo-validates the realized service level per route, estimates expected
     two-stage recourse cost, and auto-rebuilds capacity when the target service
     level (or plain feasibility) isn't met — re-validated against a fresh,
     independently-seeded scenario draw.
   - `modules/value_of_information.py` — Value of the Stochastic Solution (VSS) and
     Expected Value of Perfect Information (EVPI), comparing the deterministic
     mean-demand plan, the SAA plan, and a wait-and-see (perfect-foresight) plan, all
     evaluated under the same demand scenarios.
   - `modules/routing_backend.py` — optional OSRM real-road-network distance matrix
     (falls back to haversine with an explicit warning if unreachable). Haversine
     stays the default, since the synthetic demo's coordinates don't sit on a real
     road network; OSRM is intended for the "upload your own dataset" path.
4. **Web UI + XAI** (`app.py`, `modules/xai_engine.py`) — a Streamlit dashboard with
   demand-trajectory charts, a CQR calibration diagnostics tab, stockout-risk KPIs, a
   Folium routing map (deterministic or SAA mode) with VSS/EVPI, and SHAP/LIME/occlusion
   explanations for both the XGBoost and LSTM forecasters.

## Directory Structure

```
doctoral_supply_chain/
├── requirements.txt / requirements-dev.txt
├── pytest.ini / Makefile / Dockerfile / docker-compose.yml
├── .github/workflows/test.yml   # CI: pytest on push/PR
├── data/
│   ├── __init__.py
│   ├── generate_data.py         # synthetic data generator + loader (writes CSVs here)
│   └── data_loader.py           # bring-your-own-dataset validation/loading
├── modules/
│   ├── __init__.py
│   ├── forecasting.py           # XGBoost quantile / CQR / LSTM forecasters
│   ├── metrics.py                # pinball loss, CRPS, coverage, interval width
│   ├── backtesting.py            # rolling-origin walk-forward CV
│   ├── significance.py           # Diebold-Mariano / Friedman-Nemenyi
│   ├── mlflow_tracking.py        # optional MLflow logging
│   ├── optimization.py           # CVRP-SD solver (OR-Tools), deterministic
│   ├── optimization_saa.py       # SAA CVRP-SD, recourse cost, service level
│   ├── value_of_information.py   # VSS / EVPI
│   ├── routing_backend.py        # haversine / OSRM distance matrix
│   └── xai_engine.py             # SHAP (dual-mode) + LIME + LSTM occlusion
├── scripts/
│   ├── run_backtest.py           # one-command rolling-origin backtest + plot
│   └── validate_phase_[a-d].py   # acceptance-criteria validation scripts
├── tests/                        # pytest suite
└── app.py                        # Streamlit dashboard
```

## Setup

Requires Python 3.11+ (XGBoost's `reg:quantileerror` objective needs XGBoost ≥ 2.0).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt        # or requirements-dev.txt for tests/MLflow
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
  the uncertainty band shown in the forecast chart and, in Deterministic routing mode,
  as the effective demand fed into the CVRP-SD capacity constraint.
- **Vehicle Capacity** — per-vehicle capacity used by the routing solver.
- **Routing Method** — Deterministic (upper-quantile, unchanged default) or Stochastic
  (SAA, alpha-service level): the latter exposes CQR miscoverage alpha, target service
  level, and Monte Carlo scenario count.
- **Distance Backend** — Haversine (default) or OSRM real road-network distance;
  automatically falls back to haversine with a warning if OSRM is unreachable.

### Panels

- **Demand Forecast** — actual vs. quantile-forecast demand trajectory (now
  rearrangement-corrected for quantile crossing), with an optional LSTM comparison
  model (trained on demand, cached after first click), and a CSV export.
- **Forecast Diagnostics** — CQR calibration: naive vs. calibrated empirical coverage,
  pinball loss, CRPS, interval width against the nominal target, plus a static-vs-
  adaptive online coverage replay.
- **Stockout Risk** — naive mean-demand inventory sizing vs. ML-informed (quantile-sized)
  inventory, with the relative risk reduction computed live from the current scenario.
- **Vehicle Routing** — Deterministic CVRP-SD map/export (unchanged), or Stochastic
  (SAA) mode: service level achieved, expected recourse cost, capacity auto-rebuild, and
  a VSS/EVPI expander.
- **Explainability** — dual-mode SHAP (tree path-dependent vs. interventional, with an
  explicit autocorrelation caveat) + LIME for XGBoost, plus a windowed-occlusion
  explanation for the LSTM (previously unexplained).

### Performance notes

CVRP-SD solving (OR-Tools) and SHAP/LIME explanation are cached (`@st.cache_data`) keyed
on their actual inputs, and model training is cached (`@st.cache_resource`) keyed on the
active dataset plus the scenario parameters that affect it — so unrelated widget changes
(e.g. moving the vehicle capacity slider while on the Forecast tab) no longer re-solve
routing or recompute explanations on every rerun.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Rolling-origin backtest (one command, produces a per-fold table + calibration plot):

```bash
python scripts/run_backtest.py --model cqr --folds 5
# add --mlflow to also log the run to a local MLflow tracking store
```

Or via `make`: `make test`, `make backtest`, `make run`. CI (`.github/workflows/test.yml`)
runs the pytest suite on every push/PR — deliberately just `pytest tests/`, since the
suite already uses small synthetic fixtures and never calls the OSRM network backend.

## Docker

```bash
docker build -t supply-chain-cockpit .
docker run -p 8501:8501 supply-chain-cockpit
# or: docker compose up            (app only)
# or: docker compose --profile mlflow up   (app + local MLflow UI on :5000)
```

## Design notes

- **OR-Tools only.** No Gurobi dependency/license is required; the routing solver runs
  entirely on the open-source OR-Tools CP-SAT/routing library.
- **`reg:quantileerror`, not `reg:absoluteerror`.** MAE loss only targets the median
  (q=0.5) and cannot produce an upper-bound estimate; targeting q=0.90 requires
  XGBoost's native quantile objective with `quantile_alpha`.
- **Independent forecasters.** XGBoost and LSTM are not stacked — they share an
  interface so the dashboard can compare them, not so one feeds the other.
- **Honest, checked negative results are part of the design.** Static (non-adaptive)
  CQR coverage can miss its nominal target under this dataset's non-stationarity (GVI
  shocks, trend, seasonality) — that's why the adaptive mode exists, not hidden behind
  it. VSS/EVPI are reported as computed, including if a scenario draw makes either come
  out negative or near zero.
