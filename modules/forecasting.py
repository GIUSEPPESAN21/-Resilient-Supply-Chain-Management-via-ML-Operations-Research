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

from modules.metrics import crps_from_quantiles, empirical_coverage, interval_width, pinball_loss

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


def enforce_monotonic_quantiles(q_dict: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Rearrangement fix for quantile crossing (Chernozhukov, Fernandez-Val & Galichon 2010).

    Independently-fit quantile models (one XGBoost regressor per quantile level, as
    XGBQuantileEnsemble does) offer no guarantee that q10 <= q50 <= q90 row-by-row.
    The rearrangement operator sorts each row's predicted quantile vector ascending
    and reassigns the sorted values back to the (already-sorted) quantile levels —
    this is a monotone rearrangement of the estimated quantile function and is the
    minimal-distance (in L2) monotonic correction. Applied at the point predictions
    are assembled (app.py), not inside the model classes, so the underlying models
    stay untouched and this is a pure post-processing step.
    """
    levels = sorted(q_dict.keys())
    stacked = np.stack([np.asarray(q_dict[lvl], dtype=float) for lvl in levels], axis=1)
    rearranged = np.sort(stacked, axis=1)
    return {lvl: rearranged[:, i] for i, lvl in enumerate(levels)}


class ConformalizedQuantileForecaster:
    """Conformalized Quantile Regression (Romano, Patterson & Candes 2019).

    Wraps two `QuantileXGBForecaster` base models (lower/upper quantile) plus a
    median model, and calibrates their interval on a held-out, time-respecting
    split so the resulting interval carries a marginal coverage guarantee even
    if the base quantile models are themselves miscalibrated — the correction
    absorbs that miscalibration. Additive to the existing `XGBQuantileEnsemble`
    interface: exposes `.predict_quantiles()` (uncalibrated base quantiles, same
    shape as the ensemble) plus the new `.predict_interval(features, alpha)`.
    """

    def __init__(self, lower_quantile: float = 0.1, upper_quantile: float = 0.9,
                 calib_frac: float = 0.2, adaptive: bool = False,
                 adaptive_gamma: float = 0.01, **xgb_kwargs):
        if not 0 < calib_frac < 1:
            raise ValueError("calib_frac must be in (0, 1)")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.calib_frac = calib_frac
        self.model_lo = QuantileXGBForecaster(quantile=lower_quantile, **xgb_kwargs)
        self.model_hi = QuantileXGBForecaster(quantile=upper_quantile, **xgb_kwargs)
        self.model_median = QuantileXGBForecaster(quantile=0.5, **xgb_kwargs)
        # Gibbs & Candes 2021 — Adaptive Conformal Inference (ACI): optional online
        # tracking of a running miscoverage target, useful under the GVI/climate/macro
        # non-stationarity in this dataset. Off by default; enable via adaptive=True.
        self.adaptive = adaptive
        self.adaptive_gamma = adaptive_gamma
        self._calib_scores = None
        self._n_calib = 0
        self._alpha_t = None

    def _time_respecting_split(self, features: pd.DataFrame) -> tuple[list, list]:
        """Per-customer chronological split: earliest (1 - calib_frac) of each
        customer's rows go to proper-train, the most recent calib_frac go to
        calibration — never a random split, since conformity scores must be
        computed on data the base models could not have seen and that follows
        training data in time."""
        train_idx, calib_idx = [], []
        for _, g in features.groupby("customer_id"):
            g_sorted = g.sort_values("date")
            n = len(g_sorted)
            n_calib = max(1, int(round(n * self.calib_frac)))
            n_calib = min(n_calib, n - 1) if n > 1 else 0
            idx = g_sorted.index.tolist()
            if n_calib == 0:
                train_idx.extend(idx)
            else:
                train_idx.extend(idx[: n - n_calib])
                calib_idx.extend(idx[n - n_calib:])
        return train_idx, calib_idx

    def fit(self, features: pd.DataFrame, target_col: str = "demand") -> "ConformalizedQuantileForecaster":
        train_idx, calib_idx = self._time_respecting_split(features)
        if not calib_idx:
            raise ValueError("No rows available for the calibration split — need more "
                              "history per customer or a larger calib_frac.")
        train_df = features.loc[train_idx]
        calib_df = features.loc[calib_idx]

        self.model_lo.fit(train_df, target_col)
        self.model_hi.fit(train_df, target_col)
        self.model_median.fit(train_df, target_col)

        q_lo_calib = self.model_lo.predict(calib_df)
        q_hi_calib = self.model_hi.predict(calib_df)
        y_calib = calib_df[target_col].to_numpy(dtype=float)
        # Conformity score (Romano et al. 2019, eq. 3): how far y falls outside
        # the raw [q_lo, q_hi] band, signed so positive = uncovered.
        self._calib_scores = np.maximum(q_lo_calib - y_calib, y_calib - q_hi_calib)
        self._n_calib = len(self._calib_scores)
        self._alpha_t = None
        return self

    def _conformal_quantile(self, alpha: float) -> float:
        n = self._n_calib
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(self._calib_scores, level, method="higher"))

    def predict_quantiles(self, features: pd.DataFrame) -> dict[float, np.ndarray]:
        """Uncalibrated base-model quantiles — same shape/interface as
        `XGBQuantileEnsemble.predict_quantiles` so app.py can treat both alike."""
        return {
            self.lower_quantile: self.model_lo.predict(features),
            0.5: self.model_median.predict(features),
            self.upper_quantile: self.model_hi.predict(features),
        }

    def predict_interval(self, features: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated (lower, upper) with marginal coverage ~= 1 - alpha."""
        if self._calib_scores is None:
            raise RuntimeError("Call fit() before predict_interval().")
        eff_alpha = self._alpha_t if (self.adaptive and self._alpha_t is not None) else alpha
        q_hat = self._conformal_quantile(float(np.clip(eff_alpha, 1e-3, 1 - 1e-3)))
        lower = self.model_lo.predict(features) - q_hat
        upper = self.model_hi.predict(features) + q_hat
        if self.adaptive and self._alpha_t is None:
            self._alpha_t = alpha
        return lower, upper

    def update_adaptive(self, y_true: float, lower: float, upper: float, alpha: float = 0.1) -> float:
        """Gibbs & Candes 2021 online update: nudge the running miscoverage target
        alpha_t based on whether the realized value fell inside the last predicted
        interval, so interval width adapts to non-stationary drift (GVI shocks etc.)
        instead of relying on a single fixed calibration set forever."""
        if not self.adaptive:
            raise RuntimeError("Instantiate with adaptive=True to use online updates.")
        if self._alpha_t is None:
            self._alpha_t = alpha
        err = 0.0 if (lower <= y_true <= upper) else 1.0
        self._alpha_t = float(np.clip(self._alpha_t + self.adaptive_gamma * (alpha - err), 1e-3, 1 - 1e-3))
        return self._alpha_t


def train_test_holdout_split(features: pd.DataFrame, holdout_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological per-customer split reserving the most recent `holdout_frac` of
    each customer's rows as a genuinely unseen test set — used only for honest
    coverage validation, never for fitting or calibrating a model."""
    train_idx, holdout_idx = [], []
    for _, g in features.groupby("customer_id"):
        g_sorted = g.sort_values("date")
        n = len(g_sorted)
        n_holdout = max(1, int(round(n * holdout_frac)))
        n_holdout = min(n_holdout, n - 1) if n > 1 else 0
        idx = g_sorted.index.tolist()
        if n_holdout == 0:
            train_idx.extend(idx)
        else:
            train_idx.extend(idx[: n - n_holdout])
            holdout_idx.extend(idx[n - n_holdout:])
    return features.loc[train_idx], features.loc[holdout_idx]


def run_forecast_diagnostics(features: pd.DataFrame, alpha: float = 0.1, calib_frac: float = 0.2,
                              holdout_frac: float = 0.15, target_col: str = "demand",
                              **xgb_kwargs) -> dict:
    """Fits a naive independently-fit XGBoost quantile pair (no conformal
    calibration, rearrangement-corrected for crossing) and a
    ConformalizedQuantileForecaster on the same chronological training rows,
    then scores both on a held-out slice neither model touched during fitting
    or calibration. This is what lets the "does CQR actually help" claim be
    backed by a real out-of-sample number instead of an in-sample one.
    """
    train_df, holdout_df = train_test_holdout_split(features, holdout_frac)
    y_holdout = holdout_df[target_col].to_numpy(dtype=float)
    lower_level, upper_level = alpha / 2, 1 - alpha / 2

    naive_ensemble = XGBQuantileEnsemble(quantiles=(lower_level, 0.5, upper_level), **xgb_kwargs)
    naive_ensemble.fit(train_df, target_col)
    naive_preds = enforce_monotonic_quantiles(naive_ensemble.predict_quantiles(holdout_df))
    naive_lower, naive_upper = naive_preds[lower_level], naive_preds[upper_level]

    cqr = ConformalizedQuantileForecaster(lower_quantile=lower_level, upper_quantile=upper_level,
                                           calib_frac=calib_frac, **xgb_kwargs)
    cqr.fit(train_df, target_col)
    cqr_lower, cqr_upper = cqr.predict_interval(holdout_df, alpha=alpha)

    def _score(lower, upper):
        return {
            "empirical_coverage": empirical_coverage(y_holdout, lower, upper),
            "interval_width": interval_width(lower, upper),
            "pinball_lower": pinball_loss(y_holdout, lower, lower_level),
            "pinball_upper": pinball_loss(y_holdout, upper, upper_level),
            "crps": crps_from_quantiles(
                y_holdout, {lower_level: lower, upper_level: upper}, [lower_level, upper_level]),
        }

    return {
        "n_holdout": len(y_holdout),
        "nominal_coverage": 1 - alpha,
        "naive": _score(naive_lower, naive_upper),
        "cqr": _score(cqr_lower, cqr_upper),
    }


def simulate_online_coverage(cqr: ConformalizedQuantileForecaster, holdout_df: pd.DataFrame,
                              alpha: float = 0.1, target_col: str = "demand",
                              date_col: str = "date") -> dict:
    """Replays a fitted CQR forecaster's calibration sequentially in chronological
    order over `holdout_df`, comparing the static correction (fixed calibration
    quantile) against the online adaptive (Gibbs & Candes 2021) update. Diagnoses
    whether time-series non-stationarity (trend/seasonality/GVI drift) is breaking
    the exchangeability assumption the static conformal guarantee relies on — on
    this project's synthetic data it demonstrably is (see scripts/validate_phase_a.py).
    """
    holdout_sorted = holdout_df.sort_values([date_col, "customer_id"])
    y = holdout_sorted[target_col].to_numpy(dtype=float)
    q_lo = cqr.model_lo.predict(holdout_sorted)
    q_hi = cqr.model_hi.predict(holdout_sorted)

    static_q_hat = cqr._conformal_quantile(alpha)
    static_covered, adaptive_covered = 0, 0
    alpha_t = alpha
    gamma = cqr.adaptive_gamma
    for i in range(len(y)):
        lo_s, hi_s = q_lo[i] - static_q_hat, q_hi[i] + static_q_hat
        static_covered += int(lo_s <= y[i] <= hi_s)

        q_hat_t = cqr._conformal_quantile(float(np.clip(alpha_t, 1e-3, 1 - 1e-3)))
        lo_a, hi_a = q_lo[i] - q_hat_t, q_hi[i] + q_hat_t
        covered = lo_a <= y[i] <= hi_a
        adaptive_covered += int(covered)
        err = 0.0 if covered else 1.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err), 1e-3, 1 - 1e-3))

    n = len(y)
    return {"n": n, "nominal_coverage": 1 - alpha,
            "static_coverage": static_covered / n, "adaptive_coverage": adaptive_covered / n}
