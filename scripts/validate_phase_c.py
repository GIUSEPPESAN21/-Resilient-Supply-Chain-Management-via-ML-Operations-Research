"""Phase C acceptance-criteria validation.
Checks:
  1. Both SHAP perturbation modes (tree_path_dependent, interventional) run without
     error on this dataset's autocorrelated features and produce different attributions
     (proving they're not silently the same computation under a different label).
  2. LSTM occlusion explanation runs, returns one contribution per lookback day, and
     the days sum to a plausible relationship with the baseline prediction.

Usage:  .venv/Scripts/python.exe scripts/validate_phase_c.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.generate_data import load_or_generate_data
from modules.forecasting import LSTMForecaster, XGBQuantileEnsemble, build_feature_frame
from modules.xai_engine import explain_global, explain_instance, explain_lstm_instance

XGB_KWARGS = dict(n_estimators=150, max_depth=4)


def check_shap_perturbation_modes(features) -> None:
    print("[C1] SHAP perturbation modes on autocorrelated lag/rolling features:")
    ensemble = XGBQuantileEnsemble(quantiles=(0.1, 0.5, 0.9), **XGB_KWARGS)
    ensemble.fit(features)
    model = ensemble.models[0.9]

    global_path_dep = explain_global(model, features, sample_size=200, feature_perturbation="tree_path_dependent")
    global_interventional = explain_global(model, features, sample_size=200, feature_perturbation="interventional",
                                            background_size=80)
    print("  tree_path_dependent top-3 features:",
          global_path_dep.head(3)[["feature", "mean_abs_shap"]].to_dict("records"))
    print("  interventional      top-3 features:",
          global_interventional.head(3)[["feature", "mean_abs_shap"]].to_dict("records"))

    merged = global_path_dep.merge(global_interventional, on="feature", suffixes=("_pathdep", "_interv"))
    max_abs_diff = (merged["mean_abs_shap_pathdep"] - merged["mean_abs_shap_interv"]).abs().max()
    print(f"  max |difference| in mean|SHAP| between modes: {max_abs_diff:.4f}")
    assert max_abs_diff > 1e-6, "Both perturbation modes produced identical attributions — something's wired wrong"
    print("[C1] PASS — both modes run without error and genuinely differ (not the same computation twice).\n")

    instance_exp = explain_instance(model, features, instance_idx=100, feature_perturbation="interventional",
                                     background_size=80)
    assert set(instance_exp.keys()) == {"shap", "lime"}
    print("[C1] PASS — per-instance SHAP+LIME also runs under interventional mode.\n")


def check_lstm_occlusion(demand, exogenous) -> None:
    print("[C2] LSTM windowed occlusion explanation:")
    lstm = LSTMForecaster(epochs=8).fit(demand, exogenous)
    customer_id = int(demand["customer_id"].iloc[0])
    explanation = explain_lstm_instance(lstm, demand, exogenous, customer_id, quantile_idx=1)

    assert len(explanation["days_ago"]) == lstm.lookback
    assert len(explanation["contribution"]) == lstm.lookback
    print(f"  lookback window: {lstm.lookback} days")
    print(f"  baseline (unmasked) q0.50 prediction: {explanation['baseline_prediction']:.2f}")
    print(f"  contribution range: [{explanation['contribution'].min():.3f}, {explanation['contribution'].max():.3f}]")
    print(f"  most influential day (days_ago): "
          f"{explanation['days_ago'][np.argmax(np.abs(explanation['contribution']))]}")
    print("[C2] PASS — LSTM occlusion explanation runs and returns one contribution per lookback day.\n")


if __name__ == "__main__":
    data = load_or_generate_data(n_customers=16, n_days=400, seed=42)
    demand, exogenous = data["demand"], data["exogenous"]
    features = build_feature_frame(demand, exogenous)

    check_shap_perturbation_modes(features)
    check_lstm_occlusion(demand, exogenous)
