"""Probabilistic forecast scoring: pinball loss, approximate CRPS, and interval
calibration diagnostics (empirical coverage, interval width) — used by the
"Forecast diagnostics" panel in app.py and by the rolling-origin backtester in
modules/backtesting.py.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss, averaged over observations."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def crps_from_quantiles(y_true, quantile_preds, quantile_levels) -> float:
    """Approximate CRPS via quantile-loss integration: CRPS(y) = 2 * integral_0^1
    pinball_tau(y, q_tau) dtau, evaluated by trapezoidal integration over the
    supplied quantile levels (Gneiting & Raftery 2007; Laio & Tamea 2007 give
    the equivalent pinball-integral identity used here).

    `quantile_preds` is either a {level: array} dict or a 2D array of shape
    (n_obs, n_levels) aligned with `quantile_levels`.
    """
    y_true = np.asarray(y_true, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float)
    order = np.argsort(levels)
    levels_sorted = levels[order]

    if isinstance(quantile_preds, dict):
        preds = np.stack([np.asarray(quantile_preds[lvl], dtype=float) for lvl in levels], axis=1)
    else:
        preds = np.asarray(quantile_preds, dtype=float)
    preds = preds[:, order]

    diff = y_true[:, None] - preds
    pinball = np.maximum(levels_sorted[None, :] * diff, (levels_sorted[None, :] - 1) * diff)
    crps_per_obs = 2.0 * np.trapz(pinball, levels_sorted, axis=1)
    return float(crps_per_obs.mean())


def empirical_coverage(y_true, lower, upper) -> float:
    """Fraction of true values falling inside [lower, upper] — compare against the
    nominal target (e.g. 1 - alpha) to check calibration."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower, upper) -> dict:
    """Summary of interval width — catches a model that "cheats" coverage with
    absurdly wide intervals instead of being genuinely well-calibrated."""
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return {
        "mean": float(np.mean(width)),
        "std": float(np.std(width)),
        "median": float(np.median(width)),
        "p90": float(np.percentile(width, 90)),
        "min": float(np.min(width)),
        "max": float(np.max(width)),
    }
