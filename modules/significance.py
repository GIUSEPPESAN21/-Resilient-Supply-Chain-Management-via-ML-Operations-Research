"""Statistical significance testing for forecaster comparisons, so a claim like
"XGBoost beats LSTM" or "CQR beats the naive normal approximation" is backed by a
p-value rather than eyeballed charts:

  - Diebold & Mariano (1995) test, with the Harvey, Leybourne & Newbold (1997)
    small-sample correction, for comparing exactly two forecasters' loss series.
  - Friedman test + Nemenyi post-hoc (Demsar 2006, "Statistical Comparisons of
    Classifiers over Multiple Data Sets") for comparing more than two, since running
    pairwise DM tests across many forecasters would inflate the false-positive rate.

`compare_forecasters()` is the single entry point app.py / backtesting scripts should
call — it picks the right test based on how many forecasters are being compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t as t_dist


def diebold_mariano_test(loss_a, loss_b, h: int = 1) -> dict:
    """Diebold & Mariano (1995): tests whether two forecasters' loss series have
    equal predictive accuracy. `loss_a`/`loss_b` should be paired (same folds/
    observations, in the same order) — e.g. per-fold pinball loss or CRPS.
    `h` is the forecast horizon in steps (used for the autocovariance truncation in
    the long-run variance estimate); h=1 reduces to the simple variance.
    Returns the (small-sample corrected) DM statistic and a two-sided p-value under
    a Student-t(n-1) reference distribution (Harvey, Leybourne & Newbold 1997).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("Need at least 2 paired observations for the DM test.")
    d_bar = d.mean()

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        var_d += 2 * gamma_lag
    var_d /= n

    if var_d <= 0:
        # Losses identical (or numerically indistinguishable) across every fold —
        # report "no detectable difference" rather than dividing by ~0.
        return {"dm_stat": 0.0, "p_value": 1.0, "n": n, "mean_diff": float(d_bar)}

    dm_stat = d_bar / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction
    p_value = 2 * (1 - t_dist.cdf(np.abs(dm_stat_corrected), df=n - 1))
    return {"dm_stat": float(dm_stat_corrected), "p_value": float(p_value), "n": n,
            "mean_diff": float(d_bar)}


def _nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Nemenyi's q_alpha (Demsar 2006, Table 5): the studentized range critical
    value for k groups at infinite df, divided by sqrt(2)."""
    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def friedman_nemenyi_test(loss_matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test + Nemenyi post-hoc (Demsar 2006) for comparing k > 2 forecasters
    across N folds/datasets. `loss_matrix`: rows = folds, columns = forecaster names,
    values = loss (e.g. per-fold CRPS). Returns the Friedman chi-square statistic,
    its p-value, each forecaster's average rank, and the Nemenyi critical difference
    (CD) — two forecasters differ significantly if their average ranks differ by
    more than CD.
    """
    methods = loss_matrix.columns.tolist()
    k, n = len(methods), len(loss_matrix)
    if k < 3:
        raise ValueError("Friedman/Nemenyi needs >= 3 forecasters — use the Diebold-Mariano "
                          "test for exactly 2.")

    stat, p_value = friedmanchisquare(*[loss_matrix[m].to_numpy() for m in methods])

    ranks = loss_matrix.apply(lambda row: rankdata(row), axis=1, result_type="expand")
    ranks.columns = methods
    avg_ranks = ranks.mean(axis=0)

    cd = _nemenyi_q_alpha(k, alpha) * np.sqrt(k * (k + 1) / (6 * n))

    return {
        "friedman_stat": float(stat), "p_value": float(p_value),
        "avg_ranks": avg_ranks.to_dict(), "critical_difference": float(cd),
        "n_folds": n, "k_methods": k,
    }


def compare_forecasters(loss_dict: dict, alpha: float = 0.05) -> dict:
    """Single entry point: `loss_dict` maps forecaster name -> array of per-fold (or
    per-observation) losses, all the SAME length and in the SAME fold order. Runs
    Diebold-Mariano if exactly 2 forecasters are given, or Friedman + Nemenyi if
    more than 2 (running pairwise DM tests instead would inflate the false-positive
    rate — Demsar 2006's whole point).
    """
    names = list(loss_dict.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 forecasters to compare.")

    if len(names) == 2:
        a, b = names
        result = diebold_mariano_test(loss_dict[a], loss_dict[b])
        result.update({"test": "diebold_mariano", "forecasters": names,
                       "significant": bool(result["p_value"] < alpha)})
        return result

    lengths = {len(v) for v in loss_dict.values()}
    if len(lengths) != 1:
        raise ValueError("All forecasters must have the same number of folds/observations "
                          "for Friedman/Nemenyi.")
    loss_matrix = pd.DataFrame(loss_dict)
    result = friedman_nemenyi_test(loss_matrix, alpha)
    result.update({"test": "friedman_nemenyi", "forecasters": names,
                   "significant": bool(result["p_value"] < alpha)})
    return result
