"""Explainability layer: SHAP (global + local) and LIME (local) for the XGBoost forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from modules.forecasting import FEATURE_COLS


def explain_global(model, features: pd.DataFrame, sample_size: int = 300) -> pd.DataFrame:
    sample = features[FEATURE_COLS].sample(min(sample_size, len(features)), random_state=42)
    shap_values = shap.TreeExplainer(model.model).shap_values(sample)
    importance = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_instance(model, features: pd.DataFrame, instance_idx: int, num_lime_features: int = 8) -> dict:
    row = features[FEATURE_COLS].iloc[[instance_idx]]
    shap_values = shap.TreeExplainer(model.model).shap_values(row)[0]
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
