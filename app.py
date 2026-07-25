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
from modules.forecasting import (LSTMForecaster, XGBQuantileEnsemble,
                                  build_feature_frame, compute_stockout_risk)
from modules.optimization import solve_cvrp_sd, suggest_fleet_size
from modules.xai_engine import explain_global, explain_instance

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
    features = features.assign(q10=preds[0.1], q50=preds[0.5], q_target=preds[quantile_target])
    return ensemble, features


xgb_ensemble, features = train_xgb_ensemble(demand, exogenous_scenario, dataset_signature,
                                             gvi_override, quantile_target, PERF["xgb_n_estimators"])
target_model = xgb_ensemble.models[quantile_target]

baseline_capacity = features["customer_id"].map(customers.set_index("customer_id")["base_demand"])
risk_baseline = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], baseline_capacity)
risk_ml = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], features["q_target"])
risk_reduction_pct = 100 * (risk_baseline.mean() - risk_ml.mean()) / max(risk_baseline.mean(), 1e-9)


@st.cache_data(show_spinner="Solving CVRP-SD routing...")
def solve_routing_cached(depot, customers_df, demand_forecast, vehicle_capacity, num_vehicles,
                          time_limit_s):
    return solve_cvrp_sd(depot, customers_df, demand_forecast, vehicle_capacity, num_vehicles,
                          time_limit_s)


@st.cache_data(show_spinner="Computing SHAP global importance...")
def get_global_importance_cached(_model, features, sample_size):
    return explain_global(_model, features, sample_size)


@st.cache_data(show_spinner="Computing SHAP/LIME explanation...")
def get_instance_explanation_cached(_model, features, instance_idx):
    return explain_instance(_model, features, instance_idx)


tab_forecast, tab_risk, tab_routing, tab_xai = st.tabs(
    ["Demand Forecast", "Stockout Risk", "Vehicle Routing", "Explainability"]
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
            @st.cache_resource(show_spinner="Training LSTM quantile forecaster...")
            def train_lstm(_demand, _exogenous, dataset_signature, epochs):
                return LSTMForecaster(epochs=epochs).fit(_demand, _exogenous)

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
    st.subheader("CVRP-SD Vehicle Routing (hedged at ML quantile target)")
    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].sort_values("customer_id")
    q_target_by_customer = latest.set_index("customer_id")["q_target"]

    routing_customers = customers
    cap = PERF["cvrp_customer_cap"]
    if cap is not None and len(customers) > cap:
        routing_customers = customers.sample(cap, random_state=42).sort_values("customer_id").reset_index(drop=True)
        st.info(f"Fast mode: routing limited to {cap} of {len(customers)} customers — switch "
                f"to Full mode in the sidebar for the complete network.")
    demand_forecast = q_target_by_customer.reindex(routing_customers["customer_id"]).to_numpy()

    num_vehicles = suggest_fleet_size(demand_forecast, vehicle_capacity)
    st.caption(f"Planning date: {latest_date.date()} | Suggested fleet size: {num_vehicles} vehicles")

    result = solve_routing_cached(depot, routing_customers, demand_forecast, vehicle_capacity,
                                   num_vehicles, PERF["cvrp_time_limit"])
    if not result["feasible"]:
        st.warning("No feasible routing solution found for the current capacity/fleet settings.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total distance", f"{result['total_distance_km']:.1f} km")
        m2.metric("Vehicles used", result["num_vehicles_used"])
        m3.metric("Demand served", f"{result['total_demand_served']} units")

        colors = ["blue", "red", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
        fmap = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=11)
        folium.Marker([depot["lat"], depot["lon"]], popup="Depot", icon=folium.Icon(color="black")).add_to(fmap)
        for _, cust in customers.iterrows():
            folium.CircleMarker([cust["lat"], cust["lon"]], radius=4, popup=cust["name"],
                                 color="gray", fill=True).add_to(fmap)
        for route in result["routes"]:
            color = colors[route["vehicle_id"] % len(colors)]
            folium.PolyLine(route["coords"], color=color, weight=3,
                             tooltip=f"Vehicle {route['vehicle_id']} | load={route['load']} | "
                                     f"{route['distance_km']} km").add_to(fmap)
        st_folium(fmap, width=None, height=520)

        routes_rows = []
        for route in result["routes"]:
            for order, (stop_id, coord) in enumerate(zip(route["stops"], route["coords"])):
                routes_rows.append({
                    "vehicle_id": route["vehicle_id"], "stop_order": order, "customer_id": stop_id,
                    "lat": coord[0], "lon": coord[1], "route_load": route["load"],
                    "route_distance_km": route["distance_km"],
                })
        routes_df = pd.DataFrame(routes_rows)
        st.download_button("Download routing plan (CSV)", routes_df.to_csv(index=False).encode("utf-8"),
                            file_name="routes.csv", mime="text/csv")

with tab_xai:
    st.subheader("Model Explainability")
    st.caption(f"Explaining the XGBoost quantile model at q={quantile_target:.2f}.")

    global_importance = get_global_importance_cached(target_model, features, PERF["shap_sample"])
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
        explanation = get_instance_explanation_cached(target_model, features, features.index.get_loc(instance_idx))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SHAP contribution (most recent forecast)**")
            st.dataframe(explanation["shap"], use_container_width=True)
        with col_b:
            st.markdown("**LIME local explanation**")
            lime_df = pd.DataFrame(explanation["lime"], columns=["condition", "weight"])
            st.dataframe(lime_df, use_container_width=True)
