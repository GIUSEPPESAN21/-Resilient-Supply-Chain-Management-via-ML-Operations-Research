"""Explainability layer: SHAP (global + local) and LIME (local) for the XGBoost
forecaster, plus an occlusion-based explainer for the LSTM (which had no XAI at all
previously).

SHAP independence caveat (C1): this dataset's lag/rolling features (lag_1, lag_7,
lag_14, roll_mean_7, roll_std_7) are strongly autocorrelated by construction. SHAP's
two `TreeExplainer` perturbation modes handle that dependence in genuinely different,
each-imperfect ways rather than one being simply "the fix":
  - "tree_path_dependent" (this module's previous, and still default, behavior):
    approximates conditional expectations E[f(X) | X_S] using the training data's
    implicit distribution along tree paths. It respects the real correlations (never
    evaluates the model on impossible feature combinations) but can spread credit
    across correlated lag features in ways that look unstable from run to run.
  - "interventional" (opt-in here, needs a background dataset): computes textbook
    Shapley values by marginalizing with an INDEPENDENCE assumption between features.
    This satisfies the clean Shapley axioms, but because the lag features are not
    actually independent, it evaluates the model on synthetic, off-manifold
    combinations (e.g. high lag_1 with low roll_mean_7) that never occur in the real
    data — which can mislead just as easily, in a different way.
There is no independence-safe default to silently pick here; both are offered with
this caveat surfaced in the UI rather than presenting either as ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer

from modules.forecasting import FEATURE_COLS


def _tree_explainer(model, features: pd.DataFrame, feature_perturbation: str,
                     background_size: int, random_state: int = 7) -> shap.TreeExplainer:
    if feature_perturbation == "tree_path_dependent":
        return shap.TreeExplainer(model.model)
    if feature_perturbation == "interventional":
        background = features[FEATURE_COLS].sample(min(background_size, len(features)),
                                                     random_state=random_state)
        return shap.TreeExplainer(model.model, data=background, feature_perturbation="interventional")
    raise ValueError(f"Unknown feature_perturbation: {feature_perturbation!r}")


def explain_global(model, features: pd.DataFrame, sample_size: int = 300,
                    feature_perturbation: str = "tree_path_dependent",
                    background_size: int = 100) -> pd.DataFrame:
    sample = features[FEATURE_COLS].sample(min(sample_size, len(features)), random_state=42)
    explainer = _tree_explainer(model, features, feature_perturbation, background_size)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    importance = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_instance(model, features: pd.DataFrame, instance_idx: int, num_lime_features: int = 8,
                      feature_perturbation: str = "tree_path_dependent",
                      background_size: int = 100) -> dict:
    row = features[FEATURE_COLS].iloc[[instance_idx]]
    explainer = _tree_explainer(model, features, feature_perturbation, background_size)
    shap_values = explainer.shap_values(row, check_additivity=False)[0]
    shap_df = (pd.DataFrame({"feature": FEATURE_COLS, "shap_value": shap_values,
                              "feature_value": row.iloc[0].to_numpy()})
               .sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True))

    X = features[FEATURE_COLS].to_numpy()
    lime_explainer = LimeTabularExplainer(
        X, feature_names=FEATURE_COLS, mode="regression", discretize_continuous=True, random_state=42
    )
    lime_exp = lime_explainer.explain_instance(X[instance_idx], model.model.predict,
                                                num_features=num_lime_features)

    return {"shap": shap_df, "lime": lime_exp.as_list()}


def explain_lstm_instance(lstm_model, demand: pd.DataFrame, exogenous: pd.DataFrame,
                           customer_id: int, quantile_idx: int = 1) -> dict:
    """Windowed occlusion explanation for the LSTM (Bento et al. 2021 TimeSHAP
    inspiration, simplified to per-day masking rather than full Shapley-over-time —
    a "WindowSHAP"-style approximation, deliberately simpler to implement correctly
    than exact Shapley values over a 14-step sequence).

    For each of the `lookback` past days, zeroes out that day's (already
    standardized) inputs — the model's learned "no information" baseline — and
    reads off the resulting change in the chosen quantile's prediction. Coarser
    granularity than full TimeSHAP (per-day, not per-day-per-feature), but it is
    exact for what it measures: the actual effect of removing that day's information
    from this specific forward pass, not an approximation of a Shapley value.
    """
    merged = demand.merge(exogenous, on="date").sort_values(["customer_id", "date"])
    g = merged[merged["customer_id"] == customer_id]
    feats = g[lstm_model.SEQ_FEATURES].to_numpy(dtype=np.float32)
    if len(feats) < lstm_model.lookback:
        raise ValueError(f"Not enough history for customer {customer_id} to build a "
                          f"{lstm_model.lookback}-day lookback window.")
    window = feats[-lstm_model.lookback:]
    window_norm = (window - lstm_model.mean_) / lstm_model.std_

    lstm_model.net.eval()
    with torch.no_grad():
        base_input = torch.tensor(window_norm[None, :, :], dtype=torch.float32)
        baseline_pred = lstm_model.net(base_input).numpy()[0]

        contributions = np.zeros(lstm_model.lookback)
        for t in range(lstm_model.lookback):
            perturbed = window_norm.copy()
            perturbed[t, :] = 0.0  # mask day t -> standardized mean, i.e. "no information"
            perturbed_input = torch.tensor(perturbed[None, :, :], dtype=torch.float32)
            perturbed_pred = lstm_model.net(perturbed_input).numpy()[0]
            contributions[t] = baseline_pred[quantile_idx] - perturbed_pred[quantile_idx]

    days_ago = np.arange(lstm_model.lookback, 0, -1)
    return {
        "days_ago": days_ago,
        "contribution": contributions,
        "baseline_prediction": float(baseline_pred[quantile_idx]),
        "quantile": lstm_model.QUANTILES[quantile_idx],
    }
