"""Supply chain resilience cockpit: probabilistic forecasting + CVRP-SD routing + XAI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from data.generate_data import load_or_generate_data
from modules.forecasting import (LSTMForecaster, XGBQuantileEnsemble,
                                  build_feature_frame, compute_stockout_risk)
from modules.optimization import solve_cvrp_sd, suggest_fleet_size
from modules.xai_engine import explain_global, explain_instance

st.set_page_config(page_title="Supply Chain Resilience Cockpit", layout="wide")
st.title("Resilient Supply Chain Cockpit")
st.caption("Probabilistic demand forecasting (XGBoost quantile / LSTM) hedging a "
           "CVRP-SD vehicle routing plan, with SHAP/LIME explainability.")


@st.cache_data(show_spinner="Loading logistics network data...")
def get_data():
    return load_or_generate_data()


data = get_data()
demand, exogenous = data["demand"], data["exogenous"]
customers, lead_times, depot = data["customers"], data["lead_times"], data["depot"]

st.sidebar.header("Scenario Controls")
gvi_override = st.sidebar.slider("Geopolitical Volatility Index", 0, 100,
                                  int(exogenous["gvi"].iloc[-1]),
                                  help="Overrides the last 30 days of GVI to simulate a shock.")
quantile_target = st.sidebar.slider("ML Quantile Target", 0.50, 0.99, 0.90, step=0.01,
                                     help="Upper-bound demand quantile used to size routing capacity.")
vehicle_capacity = st.sidebar.slider("Vehicle Capacity (units)", 200, 1200, 600, step=50)

exogenous_scenario = exogenous.copy()
exogenous_scenario.loc[exogenous_scenario.index[-30:], "gvi"] = gvi_override


@st.cache_resource(show_spinner="Training XGBoost quantile ensemble...")
def train_xgb_ensemble(_demand, _exogenous, gvi_override, quantile_target):
    features = build_feature_frame(_demand, _exogenous)
    ensemble = XGBQuantileEnsemble(quantiles=(0.1, 0.5, quantile_target))
    ensemble.fit(features)
    preds = ensemble.predict_quantiles(features)
    features = features.assign(q10=preds[0.1], q50=preds[0.5], q_target=preds[quantile_target])
    return ensemble, features


xgb_ensemble, features = train_xgb_ensemble(demand, exogenous_scenario, gvi_override, quantile_target)
target_model = xgb_ensemble.models[quantile_target]

baseline_capacity = features["customer_id"].map(customers.set_index("customer_id")["base_demand"])
risk_baseline = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], baseline_capacity)
risk_ml = compute_stockout_risk(features["q10"], features["q50"], features["q_target"], features["q_target"])
risk_reduction_pct = 100 * (risk_baseline.mean() - risk_ml.mean()) / max(risk_baseline.mean(), 1e-9)

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

    with st.expander("Compare against LSTM quantile forecaster"):
        if st.button("Train LSTM comparison model"):
            @st.cache_resource(show_spinner="Training LSTM quantile forecaster...")
            def train_lstm(_demand, _exogenous):
                return LSTMForecaster(epochs=15).fit(_demand, _exogenous)

            lstm = train_lstm(demand, exogenous_scenario)
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
    st.dataframe(risk_by_customer.rename(columns={
        "name": "Customer", "risk_baseline": "Baseline risk", "risk_ml": "ML-informed risk"
    })[["Customer", "Baseline risk", "ML-informed risk"]], use_container_width=True)

with tab_routing:
    st.subheader("CVRP-SD Vehicle Routing (hedged at ML quantile target)")
    latest_date = features["date"].max()
    latest = features[features["date"] == latest_date].sort_values("customer_id")
    demand_forecast = latest.set_index("customer_id")["q_target"].reindex(customers["customer_id"]).to_numpy()

    num_vehicles = suggest_fleet_size(demand_forecast, vehicle_capacity)
    st.caption(f"Planning date: {latest_date.date()} | Suggested fleet size: {num_vehicles} vehicles")

    result = solve_cvrp_sd(depot, customers, demand_forecast, vehicle_capacity, num_vehicles)
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

with tab_xai:
    st.subheader("Model Explainability")
    st.caption(f"Explaining the XGBoost quantile model at q={quantile_target:.2f}.")

    global_importance = explain_global(target_model, features)
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
        explanation = explain_instance(target_model, features, features.index.get_loc(instance_idx))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SHAP contribution (most recent forecast)**")
            st.dataframe(explanation["shap"], use_container_width=True)
        with col_b:
            st.markdown("**LIME local explanation**")
            lime_df = pd.DataFrame(explanation["lime"], columns=["condition", "weight"])
            st.dataframe(lime_df, use_container_width=True)
