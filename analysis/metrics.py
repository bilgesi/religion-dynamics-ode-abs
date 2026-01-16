"""
Statistical metrics for model evaluation.

Computes MAE, R-squared, and ODE-ABS comparison metrics for time series data.
Analysis module.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def extract_total_share(out: Dict, religions: List[int]) -> Dict[int, np.ndarray]:
    """
    Total share (B+M+P) per religion divided by N.
    """
    t = np.array(out["t"], dtype=float)
    S = np.array(out["S"], dtype=float)
    totals = {}
    N = S.copy()
    for r in religions:
        Br = np.array(out["B"][str(r)], dtype=float)
        Mr = np.array(out["M"][str(r)], dtype=float)
        Pr = np.array(out["P"][str(r)], dtype=float)
        N = N + Br + Mr + Pr
    for r in religions:
        Br = np.array(out["B"][str(r)], dtype=float)
        Mr = np.array(out["M"][str(r)], dtype=float)
        Pr = np.array(out["P"][str(r)], dtype=float)
        totals[r] = (Br + Mr + Pr) / np.maximum(N, 1e-12)
    return totals


def compare_ode_abs_totals(ode_out: Dict, abs_out: Dict, religions: List[int]) -> Dict:
    """
    Compute MAE and R^2 for each religion's total share time series.
    Returns per-religion metrics + pooled metrics.
    """
    ode_tot = extract_total_share(ode_out, religions)
    abs_tot = extract_total_share(abs_out, religions)

    per_r = {}
    maes = []
    r2s = []
    for r in religions:
        m = mae(ode_tot[r], abs_tot[r])
        rr = r2(ode_tot[r], abs_tot[r])
        per_r[str(r)] = {"mae": m, "r2": rr}
        maes.append(m)
        r2s.append(rr)

    return {
        "per_religion": per_r,
        "mean_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_r2": float(np.mean(r2s)) if r2s else 0.0,
    }


def _align_series(t_ref: np.ndarray, t_other: np.ndarray, y_other: np.ndarray) -> np.ndarray:
    """
    Interpolate y_other(t_other) onto t_ref if grids differ.
    """
    if len(t_ref) == len(t_other) and np.allclose(t_ref, t_other):
        return y_other
    return np.interp(t_ref, t_other, y_other)


def per_religion_total_share_metrics(ode_out: dict, abs_out: dict, religions: list[int]) -> dict:
    """
    For each religion r: compare total share (B+M+P)/N between ODE (solid) and ABS mean (dashed).
    Returns {r: {mae, rmse, r2}, ...} plus summary.
    """
    t_ode = np.array(ode_out["t"], dtype=float)
    t_abs = np.array(abs_out["t"], dtype=float)

    # build N(t) for ODE and ABS
    S_ode = np.array(ode_out["S"], dtype=float)
    N_ode = S_ode.copy()
    for r in religions:
        N_ode += np.array(ode_out["B"][str(r)], dtype=float)
        N_ode += np.array(ode_out["M"][str(r)], dtype=float)
        N_ode += np.array(ode_out["P"][str(r)], dtype=float)

    S_abs = np.array(abs_out["S"], dtype=float)
    N_abs = S_abs.copy()
    for r in religions:
        N_abs += np.array(abs_out["B"][str(r)], dtype=float)
        N_abs += np.array(abs_out["M"][str(r)], dtype=float)
        N_abs += np.array(abs_out["P"][str(r)], dtype=float)

    out = {}
    maes, rmses, r2s = [], [], []

    for r in religions:
        y_ode = (
            np.array(ode_out["B"][str(r)], dtype=float)
            + np.array(ode_out["M"][str(r)], dtype=float)
            + np.array(ode_out["P"][str(r)], dtype=float)
        ) / np.maximum(N_ode, 1e-12)

        y_abs = (
            np.array(abs_out["B"][str(r)], dtype=float)
            + np.array(abs_out["M"][str(r)], dtype=float)
            + np.array(abs_out["P"][str(r)], dtype=float)
        ) / np.maximum(N_abs, 1e-12)

        # align ABS onto ODE time grid (safe)
        y_abs_aligned = _align_series(t_ode, t_abs, y_abs)

        m = mae(y_ode, y_abs_aligned)
        e = rmse(y_ode, y_abs_aligned)
        rr = r2(y_ode, y_abs_aligned)

        out[str(r)] = {"mae": m, "rmse": e, "r2": rr}
        maes.append(m); rmses.append(e); r2s.append(rr)

    return {
        "per_religion": out,
        "summary": {
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "max_mae": float(np.max(maes)),
            "max_rmse": float(np.max(rmses)),
            "min_r2": float(np.min(r2s)),
        },
    }
