"""Probabilistic demand forecasting: XGBoost quantile regression + LSTM quantile head.

Both forecasters expose comparable interfaces so `app.py` can select or compare them.
Target quantile defaults to q=0.90 (upper-bound demand for capacity/risk hedging), using
XGBoost's native `reg:quantileerror` objective — `reg:absoluteerror` only fits the
median and cannot target an arbitrary quantile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import norm

# Streamlit's local file watcher walks every imported module's __path__ to detect
# changes; torch registers a custom __path__._path on torch.classes that isn't a
# real filesystem path, so the walk raises. The exception is caught and logged by
# Streamlit (harmless), but silencing it here keeps the __path__ empty so the
# watcher has nothing to trip over.
torch.classes.__path__ = []

FEATURE_COLS = ["customer_id", "dow", "month", "lag_1", "lag_7", "lag_14",
                 "roll_mean_7", "roll_std_7", "gvi", "climate_index", "macro_index"]


def build_feature_frame(demand: pd.DataFrame, exogenous: pd.DataFrame) -> pd.DataFrame:
    df = demand.merge(exogenous, on="date").sort_values(["customer_id", "date"]).reset_index(drop=True)
    grp = df.groupby("customer_id")["demand"]
    df["lag_1"] = grp.shift(1)
    df["lag_7"] = grp.shift(7)
    df["lag_14"] = grp.shift(14)
    df["roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_std_7"] = grp.transform(lambda s: s.shift(1).rolling(7).std())
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.dropna().reset_index(drop=True)


class QuantileXGBForecaster:
    """Single-quantile XGBoost regressor using the native quantile-error objective."""

    def __init__(self, quantile: float = 0.9, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.quantile = quantile
        self.model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, features: pd.DataFrame, target_col: str = "demand") -> "QuantileXGBForecaster":
        self.model.fit(features[FEATURE_COLS], features[target_col])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features[FEATURE_COLS])

    def set_quantile(self, quantile: float) -> None:
        """Changes the target quantile; caller must re-fit afterwards."""
        self.quantile = quantile
        self.model.set_params(quantile_alpha=quantile)


class XGBQuantileEnsemble:
    """Fits low/median/high quantiles together to expose an uncertainty band."""

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9), **kwargs):
        self.quantiles = quantiles
        self.models = {q: QuantileXGBForecaster(quantile=q, **kwargs) for q in quantiles}

    def fit(self, features: pd.DataFrame, target_col: str = "demand") -> "XGBQuantileEnsemble":
        for model in self.models.values():
            model.fit(features, target_col)
        return self

    def predict_quantiles(self, features: pd.DataFrame) -> dict[float, np.ndarray]:
        return {q: model.predict(features) for q, model in self.models.items()}


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_quantiles: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMForecaster:
    """Sequence model predicting demand quantiles (0.1/0.5/0.9) via pinball loss."""

    QUANTILES = (0.1, 0.5, 0.9)
    SEQ_FEATURES = ["demand", "gvi", "climate_index", "macro_index"]

    def __init__(self, lookback: int = 14, hidden_size: int = 32, epochs: int = 15,
                 lr: float = 1e-3, random_state: int = 42):
        self.lookback = lookback
        torch.manual_seed(random_state)
        self.net = _LSTMNet(len(self.SEQ_FEATURES), hidden_size, len(self.QUANTILES))
        self.epochs = epochs
        self.lr = lr
        self.mean_ = None
        self.std_ = None

    def _sequences(self, demand: pd.DataFrame, exogenous: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        merged = demand.merge(exogenous, on="date").sort_values(["customer_id", "date"])
        seqs, targets = [], []
        for _, g in merged.groupby("customer_id"):
            feats = g[self.SEQ_FEATURES].to_numpy(dtype=np.float32)
            for i in range(self.lookback, len(feats)):
                seqs.append(feats[i - self.lookback:i])
                targets.append(feats[i, 0])
        return np.stack(seqs), np.array(targets, dtype=np.float32)

    def fit(self, demand: pd.DataFrame, exogenous: pd.DataFrame) -> "LSTMForecaster":
        X, y = self._sequences(demand, exogenous)
        self.mean_ = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self.std_ = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
        X_t = torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        q_t = torch.tensor(self.QUANTILES, dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            preds = self.net(X_t)
            errors = y_t.unsqueeze(1) - preds
            loss = torch.maximum(q_t * errors, (q_t - 1) * errors).mean()
            loss.backward()
            opt.step()
        return self

    def predict_quantiles(self, demand: pd.DataFrame, exogenous: pd.DataFrame) -> np.ndarray:
        X, _ = self._sequences(demand, exogenous)
        X_t = torch.tensor((X - self.mean_) / self.std_, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(X_t).numpy()

    def predict(self, demand: pd.DataFrame, exogenous: pd.DataFrame, quantile: float = 0.9) -> np.ndarray:
        preds = self.predict_quantiles(demand, exogenous)
        idx = min(range(len(self.QUANTILES)), key=lambda i: abs(self.QUANTILES[i] - quantile))
        return preds[:, idx]


def compute_stockout_risk(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray,
                           capacity: float | np.ndarray) -> np.ndarray:
    """Approximates P(demand > capacity) per row from a normal fit to the (q10, q50, q90) band."""
    q10, q50, q90 = np.asarray(q10), np.asarray(q50), np.asarray(q90)
    std = np.clip((q90 - q10) / (2 * 1.2816), 1e-6, None)
    return np.clip(1 - norm.cdf(capacity, loc=q50, scale=std), 0, 1)
