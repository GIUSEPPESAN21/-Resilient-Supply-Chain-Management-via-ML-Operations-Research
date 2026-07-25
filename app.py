"""Supply chain resilience cockpit: probabilistic forecasting + CVRP-SD routing + XAI."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from data.data_loader import (build_uploaded_dataset, centroid_depot, template_csv_bytes,
                               validate_uploaded_data)
from data.generate_data import load_or_generate_data
from modules.forecasting import (ConformalizedQuantileForecaster, LSTMForecaster,
                                  XGBQuantileEnsemble, build_feature_frame,
                                  compute_stockout_risk, enforce_monotonic_quantiles,
                                  run_forecast_diagnostics, simulate_online_coverage,
                                  train_test_holdout_split)
from modules.optimization import solve_cvrp_sd, suggest_fleet_size
from modules.optimization_saa import sample_demand_scenarios, solve_saa_cvrp_sd_target_capacity
from modules.routing_backend import get_distance_matrix
from modules.value_of_information import compute_vss_evpi_report
from modules.xai_engine import explain_global, explain_instance, explain_lstm_instance

st.set_page_config(page_title="Supply Chain Resilience Cockpit", layout="wide")
st.title("Resilient Supply Chain Cockpit")
st.caption("Probabilistic demand forecasting (XGBoost quantile / LSTM) hedging a "
           "CVRP-SD vehicle routing plan, with SHAP/LIME explainability.")


# --------------------------------------------------------------------------------------
# Sidebar: data source (synthetic demo or your own dataset) + performance mode
# --------------------------------------------------------------------------------------

def _read_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))


st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Data source", ["Synthetic demo", "Upload my own dataset"], index=0,
    help="Bring your own logistics data (CSV) and run the same forecast/risk/routing/"
         "XAI pipeline against it.")

sim_customers, sim_days = 24, 730
customers_file = demand_file = exogenous_file = None
depot_lat, depot_lon = 4.7110, -74.0721

if data_source == "Upload my own dataset":
    with st.sidebar.expander("Required CSV schema", expanded=False):
        st.markdown(
            "- **customers.csv** (required): `customer_id, name, lat, lon, base_demand`\n"
            "- **demand.csv** (required): `date, customer_id, demand` — at least 15 daily "
            "rows per customer\n"
            "- **exogenous.csv** (optional): `date, gvi, climate_index, macro_index` — "
            "omit it and neutral placeholder values are used instead"
        )
        for fname, content in template_csv_bytes().items():
            st.download_button(f"Download {fname} template", content, file_name=fname,
                                mime="text/csv", key=f"template_{fname}")
    customers_file = st.sidebar.file_uploader("customers.csv", type="csv")
    demand_file = st.sidebar.file_uploader("demand.csv", type="csv")
    exogenous_file = st.sidebar.file_uploader("exogenous.csv (optional)", type="csv")

    if customers_file is not None:
        try:
            _preview = _read_upload(customers_file)
            if {"lat", "lon"}.issubset(_preview.columns):
                _default_depot = centroid_depot(_preview)
                depot_lat, depot_lon = _default_depot["lat"], _default_depot["lon"]
        except Exception:
            pass
    depot_lat = st.sidebar.number_input("Depot latitude", value=depot_lat, format="%.4f")
    depot_lon = st.sidebar.number_input("Depot longitude", value=depot_lon, format="%.4f")
else:
    st.sidebar.subheader("Simulation Size")
    sim_customers = st.sidebar.slider("Simulated customers", 5, 60, 24,
                                       help="Fewer customers = faster training/routing/SHAP "
                                            "while developing.")
    sim_days = st.sidebar.slider("Simulated history (days)", 90, 730, 730, step=30)

st.sidebar.header("Performance Mode")
perf_mode = st.sidebar.radio(
    "Performance mode", ["Fast", "Full"], index=0,
    help="Fast trims sample sizes, solve time, and model size — recommended on "
         "free-tier cloud hosting. Full uses production-quality settings.")
PERF = {
    "Fast": dict(shap_sample=50, cvrp_time_limit=2, cvrp_customer_cap=30,
                 xgb_n_estimators=150, lstm_epochs=8),
    "Full": dict(shap_sample=300, cvrp_time_limit=5, cvrp_customer_cap=None,
                 xgb_n_estimators=300, lstm_epochs=15),
}[perf_mode]


# --------------------------------------------------------------------------------------
# Load data (synthetic, cached, or uploaded + validated)
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading logistics network data...")
def get_synthetic_data(n_customers: int, n_days: int) -> dict:
    return load_or_generate_data(n_customers=n_customers, n_days=n_days)


if data_source == "Upload my own dataset":
    if customers_file is None or demand_file is None:
        st.info("Upload customers.csv and demand.csv in the sidebar to run the analysis.")
        st.stop()

    customers_df = _read_upload(customers_file)
    demand_df = _read_upload(demand_file)
    exogenous_df = _read_upload(exogenous_file) if exogenous_file is not None else None

    validation_errors = validate_uploaded_data(customers_df, demand_df, exogenous_df)
    if validation_errors:
        st.error("Your uploaded dataset has the following issue(s) — fix and re-upload:")
        for err in validation_errors:
            st.error(f"- {err}")
        st.stop()

    depot = {"customer_id": 0, "name": "Uploaded depot", "lat": depot_lat, "lon": depot_lon}
    data, load_warnings = build_uploaded_dataset(customers_df, demand_df, exogenous_df, depot)
    for warning in load_warnings:
        st.sidebar.warning(warning)
else:
    data = get_synthetic_data(sim_customers, sim_days)

demand, exogenous = data["demand"], data["exogenous"]
customers, lead_times, depot = data["customers"], data["lead_times"], data["depot"]

dataset_signature = (len(demand), len(customers), int(demand["customer_id"].nunique()),
                      str(demand["date"].min()), str(demand["date"].max()))

with st.expander(f"Dataset summary — {data_source}", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(customers))
    c2.metric("Demand rows", len(demand))
    c3.metric("Date range", f"{demand['date'].min().date()} to {demand['date'].max().date()}")


# --------------------------------------------------------------------------------------
# Scenario controls
# --------------------------------------------------------------------------------------

st.sidebar.header("Scenario Controls")
gvi_override = st.sidebar.slider("Geopolitical Volatility Index", 0, 100,
                                  int(exogenous["gvi"].iloc[-1]),
                                  help="Overrides the last 30 days of GVI to simulate a shock.")
quantile_target = st.sidebar.slider("ML Quantile Target", 0.50, 0.99, 0.90, step=0.01,
                                     help="Upper-bound demand quantile used to size routing capacity.")
vehicle_capacity = st.sidebar.slider("Vehicle Capacity (units)", 200, 1200, 600, step=50)

exogenous_scenario = exogenous.copy()
exogenous_scenario.loc[exogenous_scenario.index[-30:], "gvi"] = gvi_override

st.sidebar.header("Routing Method")
routing_mode = st.sidebar.radio(
    "Routing mode", ["Deterministic (upper-quantile)", "Stochastic (SAA, α-service level)"],
    index=0,
    help="Deterministic hedges capacity at the ML Quantile Target above (today's "
         "approach). Stochastic (SAA) instead routes on a Sample Average "
         "Approximation of the calibrated demand distribution and reports a "
         "Monte Carlo-measured service level, expected recourse cost, and VSS/EVPI.")
saa_service_level_alpha, saa_n_scenarios, saa_calib_alpha = 0.95, 200, 0.10
if routing_mode.startswith("Stochastic"):
    saa_calib_alpha = st.sidebar.slider(
        "CQR miscoverage (alpha) for demand scenarios", 0.05, 0.30, 0.10, step=0.01,
        help="Feeds the calibrated interval that Monte Carlo demand scenarios are "
             "sampled from — see the Forecast Diagnostics tab.")
    saa_service_level_alpha = st.sidebar.slider(
        "Target service level (alpha)", 0.80, 0.99, 0.95, step=0.01,
        help="Required P(route load <= capacity), measured empirically over the "
             "Monte Carlo scenarios.")
    saa_n_scenarios = st.sidebar.slider("Monte Carlo scenarios", 50, 500, 200, step=50)

st.sidebar.header("Distance Backend")
distance_backend_mode = st.sidebar.radio(
    "Distance backend", ["Haversine", "OSRM (real road network)"], index=0,
    help="Haversine is the default and the only physically meaningful choice for "
         "the synthetic demo — those coordinates are randomly placed around a "
         "depot and don't sit on a real road network. OSRM (real travel distance) "
         "is intended for the 'Upload my own dataset' path with real-world "
         "coordinates; it falls back to haversine with a warning if unreachable.")
if distance_backend_mode.startswith("OSRM") and data_source == "Synthetic demo":
    st.sidebar.caption("Note: synthetic coordinates aren't on a real road network — "
                       "OSRM distances here are for demonstration only.")


# --------------------------------------------------------------------------------------
# Cached heavy compute: XGBoost ensemble, CVRP-SD solve, SHAP/LIME
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training XGBoost quantile ensemble...")
def train_xgb_ensemble(_demand, _exogenous, dataset_signature, gvi_override, quantile_target,
                        n_estimators):
    features = build_feature_frame(_demand, _exogenous)
    ensemble = XGBQuantileEnsemble(quantiles=(0.1, 0.5, quantile_target), n_estimators=n_estimators)
    ensemble.fit(features)
    preds = ensemble.predict_quantiles(features)
    # Chernozhukov, Fernandez-Val & Galichon 2010 — rearrangement fix so q10 <= q50 <=
    # q_target holds row-by-row; the independently-fit XGBoost quantile models offer no
    # such guarantee on their own. Applied here at assembly time, not inside the models.
    preds = enforce_monotonic_quantiles(preds)
    features = features.assign(q10=preds[0.1], q50=preds[0.5], q_target=preds[quantile_target])
    return ensemble, features


@st.cache_resource(show_spinner="Training LSTM quantile forecaster...")
def train_lstm(_demand, _exogenous, dataset_signature, epochs):
    return LSTMForecaster(epochs=epochs).fit(_demand, _exogenous)


xgb_ensemble, features = train_xgb_ensemble(demand, exogenous_scenario, dataset_signature,
                                             gvi_override, quantile_target, PERF["xgb_n_estimators"])
target_model = xgb_ensemble.models[quantile_target]

baseline_capacity = features["customer_id"].map(customers.set_index("customer_id")["base_demand"])
risk_baseline = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], baseline_capacity)
risk_ml = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], features["q_target"])
risk_reduction_pct = 100 * (risk_baseline.mean() - risk_ml.mean()) / max(risk_baseline.mean(), 1e-9)


@st.cache_data(show_spinner="Solving CVRP-SD routing...")
def solve_routing_cached(depot, customers_df, demand_forecast, vehicle_capacity, num_vehicles,
                          time_limit_s, distance_matrix_km=None):
    return solve_cvrp_sd(depot, customers_df, demand_forecast, vehicle_capacity, num_vehicles,
                          time_limit_s, distance_matrix_km=distance_matrix_km)


def render_route_map(routing_result, depot, customers_df):
    colors = ["blue", "red", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
    fmap = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=11)
    folium.Marker([depot["lat"], depot["lon"]], popup="Depot", icon=folium.Icon(color="black")).add_to(fmap)
    for _, cust in customers_df.iterrows():
        folium.CircleMarker([cust["lat"], cust["lon"]], radius=4, popup=cust["name"],
                             color="gray", fill=True).add_to(fmap)
    for route in routing_result["routes"]:
        color = colors[route["vehicle_id"] % len(colors)]
        folium.PolyLine(route["coords"], color=color, weight=3,
                         tooltip=f"Vehicle {route['vehicle_id']} | load={route['load']} | "
                                 f"{route['distance_km']} km").add_to(fmap)
    st_folium(fmap, width=None, height=520)


def offer_routes_download(routing_result):
    routes_rows = []
    for route in routing_result["routes"]:
        for order, (stop_id, coord) in enumerate(zip(route["stops"], route["coords"])):
            routes_rows.append({
                "vehicle_id": route["vehicle_id"], "stop_order": order, "customer_id": stop_id,
                "lat": coord[0], "lon": coord[1], "route_load": route["load"],
                "route_distance_km": route["distance_km"],
            })
    routes_df = pd.DataFrame(routes_rows)
    st.download_button("Download routing plan (CSV)", routes_df.to_csv(index=False).encode("utf-8"),
                        file_name="routes.csv", mime="text/csv",
                        key=f"routes_dl_{routing_result.get('total_distance_km')}_{len(routes_rows)}")


@st.cache_data(show_spinner="Computing SHAP global importance...")
def get_global_importance_cached(_model, features, sample_size, feature_perturbation):
    return explain_global(_model, features, sample_size, feature_perturbation=feature_perturbation)


@st.cache_data(show_spinner="Computing SHAP/LIME explanation...")
def get_instance_explanation_cached(_model, features, instance_idx, feature_perturbation):
    return explain_instance(_model, features, instance_idx, feature_perturbation=feature_perturbation)


@st.cache_data(show_spinner="Computing LSTM occlusion explanation...")
def get_lstm_explanation_cached(_lstm_model, demand, exogenous, dataset_signature, customer_id,
                                 quantile_idx):
    return explain_lstm_instance(_lstm_model, demand, exogenous, customer_id, quantile_idx)


tab_forecast, tab_diagnostics, tab_risk, tab_routing, tab_xai = st.tabs(
    ["Demand Forecast", "Forecast Diagnostics", "Stockout Risk", "Vehicle Routing", "Explainability"]
)

with tab_forecast:
    st.subheader("Aggregate Demand Trajectory (last 120 days)")
    daily = (features.groupby("date")[["demand", "q10", "q50", "q_target"]]
             .sum().tail(120).reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["q_target"], line=dict(width=0),
                              name=f"q{quantile_target:.2f} upper bound", showlegend=False))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["q10"], fill="tonexty", line=dict(width=0),
                              name="q0.10-q_target band", fillcolor="rgba(99,110,250,0.2)"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["q50"], line=dict(color="#636efa"), name="Median forecast (q0.50)"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["demand"], line=dict(color="#EF553B", dash="dot"), name="Actual demand"))
    fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Units (all customers)")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download forecast (CSV)", daily.to_csv(index=False).encode("utf-8"),
                        file_name="forecast.csv", mime="text/csv")

    with st.expander("Compare against LSTM quantile forecaster"):
        if st.button("Train LSTM comparison model"):
            st.session_state["lstm_trained"] = True

        if st.session_state.get("lstm_trained"):
            lstm = train_lstm(demand, exogenous_scenario, dataset_signature, PERF["lstm_epochs"])
            lstm_preds = lstm.predict_quantiles(demand, exogenous_scenario)
            lstm_df = demand.merge(exogenous_scenario, on="date").sort_values(["customer_id", "date"])
            lstm_df = lstm_df.groupby("customer_id", group_keys=False).apply(lambda g: g.iloc[lstm.lookback:])
            lstm_df = lstm_df.reset_index(drop=True).assign(
                q10=lstm_preds[:, 0], q50=lstm_preds[:, 1], q90=lstm_preds[:, 2])
            lstm_daily = lstm_df.groupby("date")[["demand", "q10", "q50", "q90"]].sum().tail(120).reset_index()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=lstm_daily["date"], y=lstm_daily["q90"], line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=lstm_daily["date"], y=lstm_daily["q10"], fill="tonexty", line=dict(width=0),
                                       name="q0.10-q0.90 band", fillcolor="rgba(0,204,150,0.2)"))
            fig2.add_trace(go.Scatter(x=lstm_daily["date"], y=lstm_daily["q50"], line=dict(color="#00cc96"), name="LSTM median"))
            fig2.add_trace(go.Scatter(x=lstm_daily["date"], y=lstm_daily["demand"], line=dict(color="#EF553B", dash="dot"), name="Actual demand"))
            fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Units (all customers)")
            st.plotly_chart(fig2, use_container_width=True)

with tab_diagnostics:
    st.subheader("Conformalized Quantile Regression (CQR) — Calibration Diagnostics")
    st.caption(
        "Romano, Patterson & Candes (2019). Wraps independently-fit XGBoost lower/upper "
        "quantile models with a held-out calibration correction that targets a marginal "
        "coverage guarantee — additive to the point-quantile forecast above, not a "
        "replacement for it."
    )
    d1, d2, d3 = st.columns(3)
    diag_alpha = d1.slider("Miscoverage target (alpha)", 0.05, 0.30, 0.10, step=0.01,
                            help="Nominal coverage = 1 - alpha.")
    diag_calib_frac = d2.slider("Calibration fraction", 0.10, 0.40, 0.20, step=0.05,
                                 help="Share of each customer's (post-holdout) history used "
                                      "to compute conformity scores rather than to fit the "
                                      "base quantile models.")
    diag_holdout_frac = d3.slider("Held-out test fraction", 0.10, 0.30, 0.15, step=0.05,
                                   help="Most recent rows per customer, excluded from both "
                                        "fitting and calibration — used only to score coverage "
                                        "honestly, out of sample.")

    if st.button("Run coverage diagnostics"):
        st.session_state["diag_ran"] = True

    if st.session_state.get("diag_ran"):
        @st.cache_data(show_spinner="Fitting naive vs. CQR forecasters on a held-out split...")
        def _run_diagnostics(_features, dataset_signature, gvi_override, alpha, calib_frac,
                              holdout_frac, n_estimators):
            return run_forecast_diagnostics(_features, alpha=alpha, calib_frac=calib_frac,
                                             holdout_frac=holdout_frac, n_estimators=n_estimators)

        diag = _run_diagnostics(features, dataset_signature, gvi_override, diag_alpha,
                                 diag_calib_frac, diag_holdout_frac, PERF["xgb_n_estimators"])
        naive, cqr = diag["naive"], diag["cqr"]

        st.markdown(f"**Held-out rows:** {diag['n_holdout']} | "
                    f"**Nominal coverage target:** {diag['nominal_coverage']:.2f}")
        diag_table = pd.DataFrame({
            "Metric": ["Empirical coverage", "Coverage gap (pp)", "Mean interval width",
                       "Pinball loss (lower)", "Pinball loss (upper)", "CRPS"],
            "Naive (uncalibrated)": [
                f"{naive['empirical_coverage']:.3f}",
                f"{abs(naive['empirical_coverage'] - diag['nominal_coverage']) * 100:.1f}",
                f"{naive['interval_width']['mean']:.1f}",
                f"{naive['pinball_lower']:.2f}", f"{naive['pinball_upper']:.2f}",
                f"{naive['crps']:.2f}",
            ],
            "CQR (calibrated)": [
                f"{cqr['empirical_coverage']:.3f}",
                f"{abs(cqr['empirical_coverage'] - diag['nominal_coverage']) * 100:.1f}",
                f"{cqr['interval_width']['mean']:.1f}",
                f"{cqr['pinball_lower']:.2f}", f"{cqr['pinball_upper']:.2f}",
                f"{cqr['crps']:.2f}",
            ],
        })
        st.dataframe(diag_table, use_container_width=True, hide_index=True)

        cqr_gap_pp = abs(cqr["empirical_coverage"] - diag["nominal_coverage"]) * 100
        if cqr_gap_pp <= 3.0:
            st.success(f"CQR empirical coverage is within {cqr_gap_pp:.1f} pp of the nominal "
                       f"target — inside the ~2-3 pp acceptance band.")
        else:
            st.warning(
                f"CQR empirical coverage misses the nominal target by {cqr_gap_pp:.1f} pp on "
                f"this scenario/split — outside the ~2-3 pp band. This is a known failure mode: "
                f"static split-conformal's marginal guarantee assumes exchangeability between "
                f"the calibration and test rows, which the trend/seasonality/GVI drift in this "
                f"series can violate. Try the adaptive replay below, which is built for exactly "
                f"this case."
            )

        st.markdown("---")
        st.markdown("**Online adaptive coverage (Gibbs & Candes 2021)**")
        st.caption(
            "Replays the held-out slice in chronological order, comparing the static "
            "correction above against an online update that nudges the target miscoverage "
            "rate based on whether each new point was actually covered — designed for "
            "exactly the non-stationarity (GVI shocks, trend, seasonality) this dataset has."
        )
        if st.button("Run adaptive replay"):
            st.session_state["replay_ran"] = True

        if st.session_state.get("replay_ran"):
            @st.cache_data(show_spinner="Replaying calibration sequentially...")
            def _run_adaptive(_features, dataset_signature, gvi_override, alpha, calib_frac,
                               holdout_frac, n_estimators):
                lower_level, upper_level = alpha / 2, 1 - alpha / 2
                train_df, holdout_df = train_test_holdout_split(_features, holdout_frac)
                cqr_model = ConformalizedQuantileForecaster(
                    lower_quantile=lower_level, upper_quantile=upper_level,
                    calib_frac=calib_frac, adaptive=True, adaptive_gamma=0.05,
                    n_estimators=n_estimators)
                cqr_model.fit(train_df)
                return simulate_online_coverage(cqr_model, holdout_df, alpha=alpha)

            replay = _run_adaptive(features, dataset_signature, gvi_override, diag_alpha,
                                    diag_calib_frac, diag_holdout_frac, PERF["xgb_n_estimators"])
            r1, r2, r3 = st.columns(3)
            r1.metric("Static coverage", f"{replay['static_coverage']:.3f}",
                       help=f"Gap: {abs(replay['static_coverage'] - replay['nominal_coverage']) * 100:.1f} pp")
            r2.metric("Adaptive coverage", f"{replay['adaptive_coverage']:.3f}",
                       help=f"Gap: {abs(replay['adaptive_coverage'] - replay['nominal_coverage']) * 100:.1f} pp")
            r3.metric("Nominal target", f"{replay['nominal_coverage']:.3f}")

with tab_risk:
    st.subheader("Stockout Risk: Naive Mean-Demand Planning vs. ML-Informed Planning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline stockout risk", f"{risk_baseline.mean() * 100:.1f}%",
               help="Risk of stockout when inventory is sized at average historical demand.")
    c2.metric(f"ML-informed stockout risk (q{quantile_target:.2f})", f"{risk_ml.mean() * 100:.1f}%",
               help="Risk of stockout when inventory is sized at the quantile forecast.")
    c3.metric("Relative risk reduction", f"{risk_reduction_pct:.1f}%")
    st.caption("Computed from the fitted quantile forecast distribution on the current scenario "
               "(GVI override + quantile target) — not a fixed benchmark figure.")

    risk_by_customer = (features.assign(risk_baseline=risk_baseline, risk_ml=risk_ml)
                         .groupby("customer_id")[["risk_baseline", "risk_ml"]].mean().reset_index()
                         .merge(customers[["customer_id", "name"]], on="customer_id"))
    risk_table = risk_by_customer.rename(columns={
        "name": "Customer", "risk_baseline": "Baseline risk", "risk_ml": "ML-informed risk"
    })[["Customer", "Baseline risk", "ML-informed risk"]]
    st.dataframe(risk_table, use_container_width=True)
    st.download_button("Download stockout risk (CSV)", risk_table.to_csv(index=False).encode("utf-8"),
                        file_name="stockout_risk.csv", mime="text/csv")

with tab_routing:
    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].sort_values("customer_id")

    routing_customers = customers
    cap = PERF["cvrp_customer_cap"]
    if cap is not None and len(customers) > cap:
        routing_customers = customers.sample(cap, random_state=42).sort_values("customer_id").reset_index(drop=True)
        st.info(f"Fast mode: routing limited to {cap} of {len(customers)} customers — switch "
                f"to Full mode in the sidebar for the complete network.")

    routing_locations = pd.concat([
        pd.DataFrame([{"customer_id": depot["customer_id"], "lat": depot["lat"], "lon": depot["lon"]}]),
        routing_customers[["customer_id", "lat", "lon"]],
    ], ignore_index=True)

    @st.cache_data(show_spinner="Fetching OSRM distance matrix...")
    def get_distance_matrix_cached(lats, lons, mode):
        dist_km, mode_used, warning = get_distance_matrix(lats, lons, mode=mode)
        return dist_km, mode_used, warning

    distance_mode_requested = "osrm" if distance_backend_mode.startswith("OSRM") else "haversine"
    distance_matrix_km, distance_mode_used, distance_warning = get_distance_matrix_cached(
        routing_locations["lat"].to_numpy(), routing_locations["lon"].to_numpy(), distance_mode_requested)
    if distance_warning:
        st.warning(distance_warning)
    elif distance_mode_used == "osrm":
        st.caption("Using OSRM real road-network distances.")

    if routing_mode == "Deterministic (upper-quantile)":
        st.subheader("CVRP-SD Vehicle Routing (hedged at ML quantile target)")
        q_target_by_customer = latest.set_index("customer_id")["q_target"]
        demand_forecast = q_target_by_customer.reindex(routing_customers["customer_id"]).to_numpy()

        num_vehicles = suggest_fleet_size(demand_forecast, vehicle_capacity)
        st.caption(f"Planning date: {latest_date.date()} | Suggested fleet size: {num_vehicles} vehicles")

        result = solve_routing_cached(depot, routing_customers, demand_forecast, vehicle_capacity,
                                       num_vehicles, PERF["cvrp_time_limit"], distance_matrix_km)
        if not result["feasible"]:
            st.warning("No feasible routing solution found for the current capacity/fleet settings.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total distance", f"{result['total_distance_km']:.1f} km")
            m2.metric("Vehicles used", result["num_vehicles_used"])
            m3.metric("Demand served", f"{result['total_demand_served']} units")
            render_route_map(result, depot, customers)
            offer_routes_download(result)

    else:
        st.subheader("SAA CVRP-SD (Sample Average Approximation, alpha-service level)")
        st.caption(
            "Romano et al.'s calibrated CQR interval (Forecast Diagnostics tab) feeds Monte "
            "Carlo demand scenarios. Routes on a robust (0.75-quantile) representative demand, "
            "then measures the realized service level and expected recourse cost against those "
            "scenarios — replacing the deterministic upper-quantile hedge with a checked "
            "guarantee instead of an assumption."
        )

        @st.cache_resource(show_spinner="Fitting CQR forecaster for SAA demand scenarios...")
        def train_cqr_for_saa(_features, dataset_signature, gvi_override, calib_alpha, n_estimators):
            lower_level, upper_level = calib_alpha / 2, 1 - calib_alpha / 2
            model = ConformalizedQuantileForecaster(lower_quantile=lower_level, upper_quantile=upper_level,
                                                     calib_frac=0.2, n_estimators=n_estimators)
            model.fit(_features)
            return model

        cqr_saa = train_cqr_for_saa(features, dataset_signature, gvi_override, saa_calib_alpha,
                                     PERF["xgb_n_estimators"])
        latest_routing = latest[latest["customer_id"].isin(routing_customers["customer_id"])].sort_values("customer_id")
        saa_lower, saa_upper = cqr_saa.predict_interval(latest_routing, alpha=saa_calib_alpha)
        saa_median = cqr_saa.model_median.predict(latest_routing)
        num_vehicles = suggest_fleet_size(saa_median, vehicle_capacity)
        st.caption(f"Planning date: {latest_date.date()} | Suggested fleet size: {num_vehicles} vehicles")

        @st.cache_data(show_spinner="Solving SAA CVRP-SD and Monte Carlo-validating service level...")
        def solve_saa_cached(_depot, _customers, median, lower, upper, calib_alpha, capacity,
                              n_vehicles, time_limit, n_scenarios, service_level_alpha, _dist_matrix):
            return solve_saa_cvrp_sd_target_capacity(
                _depot, _customers, median, lower, upper, calib_alpha, capacity, n_vehicles,
                time_limit, n_scenarios, service_level_alpha, distance_matrix_km=_dist_matrix)

        saa_result = solve_saa_cached(depot, routing_customers, saa_median, saa_lower, saa_upper,
                                       saa_calib_alpha, vehicle_capacity, num_vehicles,
                                       PERF["cvrp_time_limit"], saa_n_scenarios,
                                       saa_service_level_alpha, distance_matrix_km)

        if not saa_result["feasible"]:
            st.warning("No feasible SAA routing solution found for the current capacity/fleet settings.")
        else:
            if saa_result.get("infeasible_at_nominal_capacity"):
                st.info(
                    f"Nominal capacity ({saa_result['capacity_before_adjustment']}) couldn't fit "
                    f"the representative demand across {num_vehicles} vehicle(s) at all — "
                    f"automatically rebuilt at capacity {saa_result['capacity_used']} (heuristic: "
                    f"representative load per vehicle + 20% headroom)."
                )
            elif saa_result["capacity_adjusted"]:
                st.info(
                    f"Nominal capacity ({saa_result['capacity_before_adjustment']}) missed the "
                    f"{saa_service_level_alpha:.0%} service-level target "
                    f"({saa_result['service_level_achieved_before_adjustment']:.1%} achieved) — "
                    f"automatically rebuilt at capacity {saa_result['capacity_used']} and "
                    f"re-validated against a fresh Monte Carlo draw."
                )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total distance", f"{saa_result['total_distance_km']:.1f} km")
            m2.metric("Vehicles used", saa_result["num_vehicles_used"])
            m3.metric("Service level achieved", f"{saa_result['overall_service_level_achieved']:.1%}",
                       help=f"Target: {saa_service_level_alpha:.0%}. Monte Carlo estimate over "
                            f"{saa_result['n_scenarios']} demand scenarios.")
            m4.metric("Expected recourse cost", f"{saa_result['expected_recourse_cost']:.1f}",
                       help="Mean, across scenarios, of (2x return-to-depot distance + stockout "
                            "penalty) for any route whose realized demand exceeds capacity.")
            render_route_map(saa_result, depot, customers)
            offer_routes_download(saa_result)

            with st.expander("Value of the Stochastic Solution (VSS) / Expected Value of Perfect Information (EVPI)"):
                st.caption(
                    "The single most defensible 'does the ML+OR integration matter' number this "
                    "app produces: VSS compares this plan's expected cost against a naive "
                    "mean-demand plan (both evaluated under the same demand scenarios); EVPI "
                    "compares it against a wait-and-see plan solved with perfect foresight. "
                    "Reported as-is, including if VSS or EVPI comes out negative or near zero."
                )
                if st.button("Compute VSS / EVPI (re-solves routing several times — slower)"):
                    scenarios_vss = sample_demand_scenarios(saa_median, saa_lower, saa_upper,
                                                             saa_calib_alpha, n_scenarios=saa_n_scenarios,
                                                             random_state=123)
                    vss_report = compute_vss_evpi_report(
                        depot, routing_customers, scenarios_vss, saa_result, vehicle_capacity,
                        num_vehicles, time_limit_s=PERF["cvrp_time_limit"], n_scenarios_ws=20,
                        ws_time_limit_s=1)
                    v1, v2, v3 = st.columns(3)
                    v1.metric("EEV cost (deterministic mean plan)",
                              f"{vss_report['EEV_cost']:.1f}" if vss_report['EEV_cost'] is not None else "n/a")
                    v2.metric("RP cost (this SAA plan)",
                              f"{vss_report['RP_cost']:.1f}" if vss_report['RP_cost'] is not None else "n/a")
                    v3.metric("WS cost (perfect foresight)",
                              f"{vss_report['WS_cost']:.1f}" if vss_report['WS_cost'] is not None else "n/a")
                    v4, v5 = st.columns(2)
                    if vss_report["VSS"] is not None:
                        v4.metric("VSS = EEV - RP", f"{vss_report['VSS']:.1f}",
                                  help="Value of using the stochastic solution instead of the naive "
                                       "mean-demand plan. Expected >= 0.")
                        if vss_report["VSS"] < 0:
                            st.warning("VSS is negative on this scenario draw/dataset — the SAA plan "
                                       "did not beat the naive mean-demand plan here. Reporting "
                                       "honestly rather than hiding it.")
                    if vss_report["EVPI"] is not None:
                        v5.metric("EVPI = RP - WS", f"{vss_report['EVPI']:.1f}",
                                  help="How much perfect demand foresight could still save beyond "
                                       "this SAA plan. Expected >= 0.")
                        if vss_report["EVPI"] < 0:
                            st.warning("EVPI is negative — likely Monte Carlo noise between the "
                                       "scenario batches used for the SAA plan vs. this evaluation.")

with tab_xai:
    st.subheader("XGBoost Explainability (SHAP + LIME)")
    st.caption(f"Explaining the XGBoost quantile model at q={quantile_target:.2f}.")

    perturbation_choice = st.radio(
        "SHAP perturbation mode", ["Tree path-dependent (default)", "Interventional (background dataset)"],
        index=0, horizontal=True,
        help="Neither mode is a clean fix for this dataset's autocorrelated lag/rolling "
             "features — see the caveat below.")
    feature_perturbation = ("tree_path_dependent" if perturbation_choice.startswith("Tree")
                             else "interventional")
    st.caption(
        "**Caveat:** `lag_1`, `lag_7`, `lag_14`, `roll_mean_7`, `roll_std_7` are strongly "
        "autocorrelated by construction. *Tree path-dependent* respects that correlation "
        "(never evaluates the model on impossible feature combinations) but can spread "
        "importance across correlated lags in ways that look unstable run to run. "
        "*Interventional* gives textbook Shapley values that satisfy the independence "
        "axiom cleanly, but because the lags are not actually independent, it evaluates "
        "the model on synthetic off-manifold combinations (e.g. high lag_1 with low "
        "roll_mean_7) that never occur in this data. Neither is unambiguously correct — "
        "pick based on which failure mode you'd rather reason about."
    )

    global_importance = get_global_importance_cached(target_model, features, PERF["shap_sample"],
                                                       feature_perturbation)
    fig3 = go.Figure(go.Bar(x=global_importance["mean_abs_shap"], y=global_importance["feature"],
                             orientation="h"))
    fig3.update_layout(height=400, xaxis_title="Mean |SHAP value|", yaxis_title="Feature",
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    selected_customer = st.selectbox("Customer", customers["customer_id"],
                                      format_func=lambda cid: customers.set_index("customer_id").loc[cid, "name"])
    cust_rows = features[features["customer_id"] == selected_customer].sort_values("date")
    if not cust_rows.empty:
        instance_idx = cust_rows.index[-1]
        explanation = get_instance_explanation_cached(target_model, features,
                                                        features.index.get_loc(instance_idx),
                                                        feature_perturbation)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SHAP contribution (most recent forecast)**")
            st.dataframe(explanation["shap"], use_container_width=True)
        with col_b:
            st.markdown("**LIME local explanation**")
            lime_df = pd.DataFrame(explanation["lime"], columns=["condition", "weight"])
            st.dataframe(lime_df, use_container_width=True)

    st.markdown("---")
    st.subheader("LSTM Explainability (windowed occlusion)")
    st.caption(
        "Bento et al. (2021) TimeSHAP-inspired occlusion, simplified to per-day masking "
        "rather than full Shapley-over-time (a \"WindowSHAP\"-style approximation). "
        "**Caveat:** granularity is per-day, not per-day-per-feature — coarser than full "
        "TimeSHAP — and it measures the exact effect of removing one day's information "
        "from this specific forward pass, not an approximation to a Shapley value the "
        "way TreeSHAP is."
    )
    if not st.session_state.get("lstm_trained"):
        st.info("Train the LSTM comparison model in the Demand Forecast tab first — this "
                "panel explains that trained model.")
    else:
        lstm_for_xai = train_lstm(demand, exogenous_scenario, dataset_signature, PERF["lstm_epochs"])
        lx1, lx2 = st.columns(2)
        lstm_customer = lx1.selectbox(
            "Customer (LSTM)", customers["customer_id"],
            format_func=lambda cid: customers.set_index("customer_id").loc[cid, "name"],
            key="lstm_xai_customer")
        lstm_quantile_choice = lx2.radio("Quantile", ["q0.10", "q0.50", "q0.90"], index=1,
                                          horizontal=True, key="lstm_xai_quantile")
        quantile_idx = {"q0.10": 0, "q0.50": 1, "q0.90": 2}[lstm_quantile_choice]

        try:
            lstm_explanation = get_lstm_explanation_cached(
                lstm_for_xai, demand, exogenous_scenario, dataset_signature,
                lstm_customer, quantile_idx)
            fig4 = go.Figure(go.Bar(x=lstm_explanation["days_ago"], y=lstm_explanation["contribution"]))
            fig4.update_layout(
                height=350, xaxis_title="Days before the forecast date",
                yaxis_title=f"Contribution to q{lstm_explanation['quantile']:.2f} prediction",
                xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(f"Baseline (unmasked) prediction: {lstm_explanation['baseline_prediction']:.1f} units.")
        except ValueError as exc:
            st.warning(str(exc))
