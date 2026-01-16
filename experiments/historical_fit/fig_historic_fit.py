#!/usr/bin/env python3
"""
Generate 3-panel figure for historical fits: Sweden, Turkey (JW), New Zealand.

Panel (a): Sweden - baseline + piecewise (break_year=2000)
Panel (b): Turkey (JW) - baseline only
Panel (c): New Zealand - baseline + piecewise (break_year from grid search)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def load_time_series(csv_path: Path, year_col: str, share_col: str, omit_year: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load time series from CSV file."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found. Available columns: {list(df.columns)}")
    if share_col not in df.columns:
        raise ValueError(f"Share column '{share_col}' not found. Available columns: {list(df.columns)}")
    
    years = df[year_col].to_numpy(dtype=float)
    y_obs = df[share_col].to_numpy(dtype=float)
    
    # Convert percentage to fraction if needed
    if np.nanmax(y_obs) > 1.5:
        y_obs = y_obs / 100.0
    
    # Filter out NaN and invalid values
    mask = np.isfinite(years) & np.isfinite(y_obs)
    years = years[mask]
    y_obs = y_obs[mask]
    
    # Omit specific year if requested
    if omit_year is not None:
        mask = years != omit_year
        years = years[mask]
        y_obs = y_obs[mask]
    
    # Clip to [0, 1]
    y_obs = np.clip(y_obs, 0.0, 1.0)
    
    # Sort by year
    sort_idx = np.argsort(years)
    years = years[sort_idx]
    y_obs = y_obs[sort_idx]
    
    return years, y_obs


def simulate_reduced_ode(
    t0: float,
    t1: float,
    y0: float,
    beta: float,
    rho: float,
    beta_post: Optional[float] = None,
    rho_post: Optional[float] = None,
    break_year: Optional[float] = None,
    K: float = 1.0,
    n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate reduced ODE: dy/dt = beta(t)*y*(K-y) - rho(t)*y
    
    If K=1.0, this is the standard model: dy/dt = beta*y*(1-y) - rho*y
    If K is fit, this is carrying-capacity model: dy/dt = beta*y*(K-y) - rho*y
    """
    def rhs(t, y):
        if break_year is not None and t >= break_year:
            beta_t = beta_post
            rho_t = rho_post
        else:
            beta_t = beta
            rho_t = rho
        return [beta_t * y[0] * (K - y[0]) - rho_t * y[0]]
    
    t_eval = np.linspace(t0, t1, n_points)
    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0=[y0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    
    y = np.clip(sol.y[0], 0.0, K)
    return sol.t, y


def simulate_piecewise_two_stage(
    t0: float,
    t1: float,
    y0: float,
    break_year: float,
    beta_pre: float,
    rho_pre: float,
    beta_post: float,
    rho_post: float,
    K: float = 1.0,
    n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate piecewise ODE in two stages (pre and post break_year).
    This avoids gradient issues in optimization when break_year is inside the integration.
    """
    # Pre-regime: t0 -> break_year
    tA, yA = simulate_reduced_ode(t0, break_year, y0, beta_pre, rho_pre, K=K, n_points=n_points)
    yb = float(yA[-1])  # Value at break_year from pre-regime
    
    # Post-regime: break_year -> t1 (start from pre's end value)
    tB, yB = simulate_reduced_ode(break_year, t1, yb, beta_post, rho_post, K=K, n_points=n_points)
    
    # Drop duplicate breakpoint from the first segment
    tA = tA[:-1]
    yA = yA[:-1]
    
    # Combine
    t = np.concatenate([tA, tB])
    y = np.concatenate([yA, yB])
    
    return t, y


def fit_baseline(
    years: np.ndarray,
    y_obs: np.ndarray,
    t0: float,
    t1: float,
    y0: float,
    fit_K: bool = False
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Fit baseline model (constant beta, rho).
    
    If fit_K=True, also fit carrying capacity K: dy/dt = beta*y*(K-y) - rho*y
    If fit_K=False, use K=1.0 (standard model): dy/dt = beta*y*(1-y) - rho*y
    """
    y_max = float(np.max(y_obs))
    
    if fit_K:
        # --- PATCH: objective (scaled SSE for numeric conditioning) ---
        SCALE = 1e6   # numeric conditioning (ppm scale)
        EPS = 1e-12
        
        def objective(x):
            beta, rho, K = x
            
            # allow rho=0
            if beta <= 0 or rho < 0 or K <= 0:
                return 1e30
            
            # enforce increasing from initial y0 (Turkey increases)
            # (K - y0) must be positive too
            if K <= y0:
                return 1e30
            if beta * (K - y0) <= rho:
                return 1e30
            
            try:
                t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho, K=K)
                y_pred = np.interp(years, t_sim, y_sim)
                
                # guard: no negatives
                if np.any(y_pred < -1e-12) or np.any(~np.isfinite(y_pred)):
                    return 1e30
                
                # scaled SSE (conditioning fix)
                loss = np.sum((SCALE * (y_obs - y_pred)) ** 2)
                return float(loss)
            except Exception:
                return 1e30
        
        # --- PATCH: better initial guess for K-model ---
        K_init = 1.2 * y_max          # near plateau
        rho_init = 0.0                # allow ~0 attrition
        # approximate exponential rate on small y (avoid linear diff)
        y_final = float(y_obs[-1])
        y_initial = float(y_obs[0])
        t_span = t1 - t0
        if t_span > 0 and y_initial > 0 and y_final > 0:
            r = (np.log(y_final) - np.log(y_initial)) / t_span  # per-year
            r = max(r, 1e-3)
        else:
            r = 1e-2
        
        # for small y: dy/dt ≈ y*(beta*(K-y) - rho) ~ y*(beta*K - rho)
        beta_init = r / max(K_init, 1e-12)
        x0 = np.array([beta_init, rho_init, K_init])
        
        # --- PATCH: tighter K bounds around observed scale ---
        K_lower = 1.01 * y_max
        K_upper = 3.0 * y_max   # not 20x
        bounds = [
            (1e-10, 1e3),        # beta can be large depending on K scale
            (0.0,   1.0),        # rho allow 0
            (K_lower, K_upper),
        ]
        # Try multiple starting points for better convergence
        best_loss = np.inf
        best_params = None
        best_sim = None
        
        for attempt in range(5):
            # Vary initial guess slightly
            if attempt == 0:
                x0_attempt = x0
            else:
                K_attempt = K_init * (1.0 + 0.2 * attempt)
                beta_attempt = beta_init * (1.0 + 0.1 * attempt)
                rho_attempt = rho_init * (1.0 + 0.1 * attempt)
                if beta_attempt * K_attempt <= rho_attempt:
                    beta_attempt = (rho_attempt + 0.01) / K_attempt
                x0_attempt = np.array([beta_attempt, rho_attempt, K_attempt])
            
            try:
                res = minimize(objective, x0_attempt, method="L-BFGS-B", bounds=bounds, options={'maxiter': 500})
                if res.success and res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x
            except Exception:
                continue
        
        if best_params is None:
            # Fallback to single attempt
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={'maxiter': 500})
            beta, rho, K = res.x
        else:
            beta, rho, K = best_params
        
        t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho, K=K)
        y_pred = np.interp(years, t_sim, y_sim)
        
        # Debug: check initial growth rate
        initial_slope = beta * (K - y0) - rho
        print(f"  Initial slope check: beta*(K-y0) - rho = {initial_slope:.6e} (should be > 0)")
        
        params = {"beta": float(beta), "rho": float(rho), "K": float(K)}
    else:
        # Fit beta, rho only (K=1.0)
        def objective(x):
            beta, rho = x
            if beta <= 0 or rho <= 0:
                return 1e9
            try:
                t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho, K=1.0)
                y_pred = np.interp(years, t_sim, y_sim)
                loss = np.mean((y_obs - y_pred) ** 2)
                return loss
            except Exception:
                return 1e9
        
        x0 = np.array([0.15, 0.05])
        bounds = [(1e-6, 5.0), (1e-6, 5.0)]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        
        beta, rho = res.x
        t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho, K=1.0)
        y_pred = np.interp(years, t_sim, y_sim)
        
        params = {"beta": float(beta), "rho": float(rho)}
    
    # Compute metrics
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    metrics = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    
    return params, metrics, y_pred, t_sim, y_sim


def fit_piecewise(
    years: np.ndarray,
    y_obs: np.ndarray,
    t0: float,
    t1: float,
    y0: float,
    break_year: float
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Fit piecewise model (beta, rho change at break_year)."""
    if break_year <= t0 or break_year >= t1:
        raise ValueError(f"break_year {break_year} must be between {t0} and {t1}")
    
    def objective(x):
        beta1, beta2, rho1, rho2 = x
        if beta1 <= 0 or beta2 <= 0 or rho1 <= 0 or rho2 <= 0:
            return 1e9
        try:
            t_sim, y_sim = simulate_reduced_ode(
                t0, t1, y0, beta1, rho1, 
                beta_post=beta2, rho_post=rho2, 
                break_year=break_year,
                K=1.0
            )
            y_pred = np.interp(years, t_sim, y_sim)
            loss = np.mean((y_obs - y_pred) ** 2)
            return loss
        except Exception:
            return 1e9
    
    x0 = np.array([0.2, 0.1, 0.03, 0.08])
    bounds = [(1e-6, 5.0), (1e-6, 5.0), (1e-6, 5.0), (1e-6, 5.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    
    beta1, beta2, rho1, rho2 = res.x
    t_sim, y_sim = simulate_reduced_ode(
        t0, t1, y0, beta1, rho1,
        beta_post=beta2, rho_post=rho2,
        break_year=break_year,
        K=1.0
    )
    y_pred = np.interp(years, t_sim, y_sim)
    
    params = {
        "beta_pre": float(beta1),
        "beta_post": float(beta2),
        "rho_pre": float(rho1),
        "rho_post": float(rho2)
    }
    
    # Compute metrics
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    metrics = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    
    return params, metrics, y_pred, t_sim, y_sim


def fit_piecewise_K(
    years: np.ndarray,
    y_obs: np.ndarray,
    t0: float,
    t1: float,
    y0: float,
    break_year: float,
    K: float
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Fit piecewise model with carrying capacity K (beta, rho change at break_year)."""
    if break_year <= t0 or break_year >= t1:
        raise ValueError(f"break_year {break_year} must be between {t0} and {t1}")
    
    SCALE = 1e6  # Same scale as baseline fit
    EPS = 1e-12
    
    def objective(x):
        beta1, beta2, rho1, rho2 = x
        # allow rho=0
        if beta1 <= 0 or beta2 <= 0 or rho1 < 0 or rho2 < 0:
            return 1e30
        # Ensure growth before break
        if K <= y0 or beta1 * (K - y0) <= rho1:
            return 1e30
        try:
            t_sim, y_sim = simulate_reduced_ode(
                t0, t1, y0, beta1, rho1,
                beta_post=beta2, rho_post=rho2,
                break_year=break_year,
                K=K
            )
            y_pred = np.interp(years, t_sim, y_sim)
            if np.any(y_pred < -EPS) or np.any(~np.isfinite(y_pred)):
                return 1e30
            loss = np.sum((SCALE * (y_obs - y_pred)) ** 2)
            return float(loss)
        except Exception:
            return 1e30
    
    # Initial guess: use reasonable values
    beta1_init = 0.1 / K if K > 0 else 0.1
    beta2_init = 0.05 / K if K > 0 else 0.05
    rho1_init = 0.0
    rho2_init = 0.001
    x0 = np.array([beta1_init, beta2_init, rho1_init, rho2_init])
    bounds = [(1e-10, 1e3), (1e-10, 1e3), (0.0, 1.0), (0.0, 1.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={'maxiter': 500})
    
    beta1, beta2, rho1, rho2 = res.x
    t_sim, y_sim = simulate_reduced_ode(
        t0, t1, y0, beta1, rho1,
        beta_post=beta2, rho_post=rho2,
        break_year=break_year,
        K=K
    )
    y_pred = np.interp(years, t_sim, y_sim)
    
    params = {
        "beta_pre": float(beta1),
        "beta_post": float(beta2),
        "rho_pre": float(rho1),
        "rho_post": float(rho2),
        "K": float(K)
    }
    
    # Compute metrics (unscaled for R²)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    metrics = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    
    return params, metrics, y_pred, t_sim, y_sim


def fit_nz_piecewise_rho(
    years: np.ndarray,
    y_obs: np.ndarray,
    t0: float,
    t1: float,
    y0: float,
    break_year: float
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Fit NZ piecewise model: beta constant, only rho changes (3 parameters)."""
    if break_year <= t0 or break_year >= t1:
        raise ValueError(f"break_year {break_year} must be between {t0} and {t1}")
    
    def objective(x):
        beta, rho1, rho2 = x
        if beta <= 0 or rho1 < 0 or rho2 < 0:
            return 1e30
        # ensure growth at start and post-break (optional)
        if beta * (1.0 - y0) <= rho1:
            return 1e30
        
        try:
            t_sim, y_sim = simulate_piecewise_two_stage(
                t0, t1, y0, break_year, beta, rho1, beta, rho2, K=1.0
            )
            y_pred = np.interp(years, t_sim, y_sim)
            return float(np.sum((y_obs - y_pred) ** 2))
        except Exception:
            return 1e30
    
    # Better initial guess: use baseline fit as starting point
    # Estimate beta from growth rate, rho from decay
    y_growth = (y_obs[-1] - y_obs[0]) / (years[-1] - years[0])
    beta_init = max(0.01, min(1.0, abs(y_growth) / (y0 * (1.0 - y0)) if y0 * (1.0 - y0) > 0 else 0.2))
    rho_init = max(0.001, min(0.1, abs(y_growth) * 0.1))
    
    x0 = np.array([beta_init, rho_init, rho_init * 1.1])  # Slightly different rho_pre and rho_post
    bounds = [(1e-12, 10.0), (0.0, 1.0), (0.0, 1.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    
    # Debug print
    print(f"    Debug: break_year={break_year:.1f}, success={res.success}, message={res.message}, fun={res.fun:.6e}, x={res.x}")
    
    beta, rho1, rho2 = res.x
    t_sim, y_sim = simulate_piecewise_two_stage(
        t0, t1, y0, break_year, beta, rho1, beta, rho2, K=1.0
    )
    y_pred = np.interp(years, t_sim, y_sim)
    
    params = {"beta": float(beta), "rho_pre": float(rho1), "rho_post": float(rho2)}
    
    # Compute metrics
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    metrics = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    
    return params, metrics, y_pred, t_sim, y_sim


def fit_nz_anchored_piecewise(
    years: np.ndarray,
    y_obs: np.ndarray,
    break_year: float
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NZ piecewise model with anchored re-initialization at break.
    
    Pre-regime: fit (beta_pre, rho_pre) using points before break_year
    Post-regime: fit (beta_post, rho_post) using points at/after break_year
    Use observed value at break_year as initial condition for post-regime (no continuity constraint).
    """
    # Split data at break_year
    pre_mask = years < break_year
    post_mask = years >= break_year
    
    years_pre = years[pre_mask]
    y_obs_pre = y_obs[pre_mask]
    years_post = years[post_mask]
    y_obs_post = y_obs[post_mask]
    
    if len(years_pre) < 2 or len(years_post) < 2:
        raise ValueError(f"Need at least 2 points in each regime. Pre: {len(years_pre)}, Post: {len(years_post)}")
    
    t0_pre = float(years_pre.min())
    t1_pre = float(years_pre.max())
    y0_pre = float(y_obs_pre[0])
    
    # Find observed value at break_year (or closest point)
    break_idx = np.argmin(np.abs(years - break_year))
    y0_post = float(y_obs[break_idx])
    t0_post = float(years[break_idx])  # Start post-regime from break_year
    t1_post = float(years_post.max())
    
    # Fit pre-regime
    def objective_pre(x):
        beta, rho = x
        if beta <= 0 or rho < 0:
            return 1e30
        if beta * (1.0 - y0_pre) <= rho:
            return 1e30
        try:
            t_sim, y_sim = simulate_reduced_ode(t0_pre, t1_pre, y0_pre, beta, rho, K=1.0)
            y_pred = np.interp(years_pre, t_sim, y_sim)
            return float(np.sum((y_obs_pre - y_pred) ** 2))
        except Exception:
            return 1e30
    
    # Better initial guess for pre-regime
    y_growth_pre = (y_obs_pre[-1] - y_obs_pre[0]) / (years_pre[-1] - years_pre[0]) if len(years_pre) > 1 else 0.0
    beta_init_pre = max(0.01, min(1.0, abs(y_growth_pre) / (y0_pre * (1.0 - y0_pre)) if y0_pre * (1.0 - y0_pre) > 0 else 0.2))
    rho_init_pre = max(0.001, min(0.1, abs(y_growth_pre) * 0.1))
    
    x0_pre = np.array([beta_init_pre, rho_init_pre])
    bounds_pre = [(1e-12, 10.0), (0.0, 1.0)]
    res_pre = minimize(objective_pre, x0_pre, method="L-BFGS-B", bounds=bounds_pre, options={"maxiter": 2000})
    beta_pre, rho_pre = res_pre.x
    
    # Fit post-regime
    def objective_post(x):
        beta, rho = x
        if beta <= 0 or rho < 0:
            return 1e30
        if beta * (1.0 - y0_post) <= rho:
            return 1e30
        try:
            t_sim, y_sim = simulate_reduced_ode(t0_post, t1_post, y0_post, beta, rho, K=1.0)
            y_pred = np.interp(years_post, t_sim, y_sim)
            return float(np.sum((y_obs_post - y_pred) ** 2))
        except Exception:
            return 1e30
    
    # Better initial guess for post-regime
    y_growth_post = (y_obs_post[-1] - y_obs_post[0]) / (years_post[-1] - years_post[0]) if len(years_post) > 1 else 0.0
    beta_init_post = max(0.01, min(1.0, abs(y_growth_post) / (y0_post * (1.0 - y0_post)) if y0_post * (1.0 - y0_post) > 0 else 0.2))
    rho_init_post = max(0.001, min(0.1, abs(y_growth_post) * 0.1))
    
    x0_post = np.array([beta_init_post, rho_init_post])
    bounds_post = [(1e-12, 10.0), (0.0, 1.0)]
    res_post = minimize(objective_post, x0_post, method="L-BFGS-B", bounds=bounds_post, options={"maxiter": 2000})
    beta_post, rho_post = res_post.x
    
    # Simulate combined trajectory
    t0_combined = float(years.min())
    t1_combined = float(years.max())
    
    # Pre-regime simulation: 2001 -> break_year
    t_sim_pre, y_sim_pre = simulate_reduced_ode(t0_pre, break_year, y0_pre, beta_pre, rho_pre, K=1.0)
    
    # Post-regime simulation: break_year -> 2023 (start from OBSERVED at break_year)
    t_sim_post, y_sim_post = simulate_reduced_ode(break_year, t1_combined, y0_post, beta_post, rho_post, K=1.0)
    
    # IMPORTANT: drop the last pre point (which is exactly break_year)
    t_sim_pre = t_sim_pre[:-1]
    y_sim_pre = y_sim_pre[:-1]
    
    # Combine (no duplicates now)
    t_sim_combined = np.concatenate([t_sim_pre, t_sim_post])
    y_sim_combined = np.concatenate([y_sim_pre, y_sim_post])
    
    # Predict at observed years
    y_pred = np.interp(years, t_sim_combined, y_sim_combined)
    
    # Quick validation: y_pred@break_year should equal y_obs@break_year (anchor check)
    break_idx = np.argmin(np.abs(years - break_year))
    if abs(years[break_idx] - break_year) < 0.1:  # If break_year matches an observed year
        print(f"    Anchor check: y_obs@{break_year:.1f} = {y_obs[break_idx]:.6f}, y_pred@{break_year:.1f} = {y_pred[break_idx]:.6f}")
    
    params = {
        "beta_pre": float(beta_pre),
        "rho_pre": float(rho_pre),
        "beta_post": float(beta_post),
        "rho_post": float(rho_post)
    }
    
    # Compute metrics
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    metrics = {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}
    
    return params, metrics, y_pred, t_sim_combined, y_sim_combined


def detect_share_column(df: pd.DataFrame) -> str:
    """Detect share column (percentage or numeric share)."""
    for col in df.columns:
        col_lower = col.lower()
        if "%" in col or "percent" in col_lower or "pct" in col_lower:
            return col
        if "share" in col_lower:
            return col
    
    # Try numeric columns (skip year column)
    year_cols = ["year", "ar", "år"]
    for col in reversed(df.columns):
        if col.lower() not in year_cols and df[col].dtype in [np.float64, np.int64]:
            return col
    
    raise ValueError(f"Could not detect share column. Columns: {list(df.columns)}")


def detect_year_column(df: pd.DataFrame) -> str:
    """Detect year column."""
    for col in df.columns:
        col_lower = col.lower()
        if "year" in col_lower or "år" in col_lower or "ar" in col_lower:
            return col
    
    raise ValueError(f"Could not detect year column. Columns: {list(df.columns)}")


def main():
    base_dir = Path(".")
    
    # Panel (a): Sweden - baseline + piecewise
    print("Loading Sweden data...")
    sweden_path = base_dir / "data" / "sweden" / "swedish_church_membership_1972_2024.csv"
    if not sweden_path.exists():
        sweden_path = base_dir / "outputs" / "runs" / "swedish_church_membership_1972_2024.csv"
    
    if not sweden_path.exists():
        print(f"ERROR: Sweden CSV not found. Tried:", file=sys.stderr)
        print(f"  - {base_dir / 'data' / 'sweden' / 'swedish_church_membership_1972_2024.csv'}", file=sys.stderr)
        print(f"  - {base_dir / 'outputs' / 'runs' / 'swedish_church_membership_1972_2024.csv'}", file=sys.stderr)
        sys.exit(1)
    
    try:
        df_sweden = pd.read_csv(sweden_path, encoding='utf-8-sig')
    except Exception:
        df_sweden = pd.read_csv(sweden_path, encoding='utf-8')
    
    sweden_year_col = detect_year_column(df_sweden)
    sweden_share_col = detect_share_column(df_sweden)
    
    years_sweden, y_obs_sweden = load_time_series(sweden_path, sweden_year_col, sweden_share_col, omit_year=1981)
    t0_sweden = float(years_sweden.min())
    t1_sweden = float(years_sweden.max())
    y0_sweden = float(y_obs_sweden[0])
    
    print(f"Sweden: {len(years_sweden)} points ({int(t0_sweden)}-{int(t1_sweden)})")
    
    # Fit Sweden - baseline
    print("Fitting Sweden baseline...")
    params_sweden_base, metrics_sweden_base, y_pred_sweden_base, t_sim_sweden_base, y_sim_sweden_base = fit_baseline(
        years_sweden, y_obs_sweden, t0_sweden, t1_sweden, y0_sweden
    )
    
    # Fit Sweden - piecewise
    print("Fitting Sweden piecewise...")
    break_year_sweden = 2000.0
    params_sweden_piece, metrics_sweden_piece, y_pred_sweden_piece, t_sim_sweden_piece, y_sim_sweden_piece = fit_piecewise(
        years_sweden, y_obs_sweden, t0_sweden, t1_sweden, y0_sweden, break_year_sweden
    )
    
    # Panel (b): Turkey (JW) - baseline only
    print("\nLoading Turkey (JW) data...")
    turkey_path = base_dir / "data" / "jw" / "timeseries" / "jw_timeseries_turkey_2014_2024.csv"
    if not turkey_path.exists():
        print(f"ERROR: Turkey CSV not found: {turkey_path}", file=sys.stderr)
        sys.exit(1)
    
    years_turkey, y_obs_turkey = load_time_series(turkey_path, "year", "share_avg")
    t0_turkey = float(years_turkey.min())
    t1_turkey = float(years_turkey.max())
    y0_turkey = float(y_obs_turkey[0])
    
    print(f"Turkey: {len(years_turkey)} points ({int(t0_turkey)}-{int(t1_turkey)})")
    print(f"  Share range: {y_obs_turkey.min():.6f} - {y_obs_turkey.max():.6f}")
    
    # Fit Turkey - baseline with carrying capacity
    print("Fitting Turkey baseline (with carrying capacity K)...")
    params_turkey, metrics_turkey, y_pred_turkey, t_sim_turkey, y_sim_turkey = fit_baseline(
        years_turkey, y_obs_turkey, t0_turkey, t1_turkey, y0_turkey, fit_K=True
    )
    K_turkey = params_turkey.get('K', 1.0)
    print(f"  Fitted K = {K_turkey:.6f}")
    print(f"  R² = {metrics_turkey['R2']:.3f}, MAE = {metrics_turkey['MAE']:.2e}")
    
    # Fit Turkey - piecewise with K (grid search for best break_year)
    print("Fitting Turkey piecewise (grid search for break_year)...")
    break_years_candidates = np.linspace(2017.0, 2021.0, 9)  # 2017, 2017.5, ..., 2021
    best_r2_turkey_piece = -np.inf
    best_break_year_turkey = None
    best_params_turkey_piece = None
    best_metrics_turkey_piece = None
    best_y_pred_turkey_piece = None
    best_t_sim_turkey_piece = None
    best_y_sim_turkey_piece = None
    
    for break_year_candidate in break_years_candidates:
        if break_year_candidate <= t0_turkey or break_year_candidate >= t1_turkey:
            continue
        try:
            # Use same K from baseline fit
            params_piece, metrics_piece, y_pred_piece, t_sim_piece, y_sim_piece = fit_piecewise_K(
                years_turkey, y_obs_turkey, t0_turkey, t1_turkey, y0_turkey, break_year_candidate, K_turkey
            )
            if metrics_piece['R2'] > best_r2_turkey_piece:
                best_r2_turkey_piece = metrics_piece['R2']
                best_break_year_turkey = break_year_candidate
                best_params_turkey_piece = params_piece
                best_metrics_turkey_piece = metrics_piece
                best_y_pred_turkey_piece = y_pred_piece
                best_t_sim_turkey_piece = t_sim_piece
                best_y_sim_turkey_piece = y_sim_piece
        except Exception:
            continue
    
    if best_break_year_turkey is not None:
        params_turkey_piece = best_params_turkey_piece
        metrics_turkey_piece = best_metrics_turkey_piece
        y_pred_turkey_piece = best_y_pred_turkey_piece
        t_sim_turkey_piece = best_t_sim_turkey_piece
        y_sim_turkey_piece = best_y_sim_turkey_piece
        print(f"  Best break_year = {best_break_year_turkey:.1f}, R² = {best_r2_turkey_piece:.3f}")
    else:
        # Fallback to baseline
        params_turkey_piece = params_turkey
        metrics_turkey_piece = metrics_turkey
        y_pred_turkey_piece = y_pred_turkey
        t_sim_turkey_piece = t_sim_turkey
        y_sim_turkey_piece = y_sim_turkey
        best_break_year_turkey = None
    
    # Save Turkey predictions and summary (for split figure script)
    print("Saving Turkey fit results...")
    turkey_pred_df = pd.DataFrame({
        "year": years_turkey,
        "obs_share": y_obs_turkey,
        "pred_baseline": y_pred_turkey,
        "pred_piecewise": y_pred_turkey_piece
    })
    turkey_pred_path = base_dir / "outputs" / "runs" / "jw_turkey_predictions.csv"
    turkey_pred_path.parent.mkdir(parents=True, exist_ok=True)
    turkey_pred_df.to_csv(turkey_pred_path, index=False)
    print(f"  Saved: {turkey_pred_path}")
    
    turkey_summary = {
        "dataset": "Turkey: Jehovah's Witnesses (2014-2024)",
        "baseline": {
            "K": float(K_turkey),
            "beta": float(params_turkey.get("beta", 0)),
            "rho": float(params_turkey.get("rho", 0)),
            "R2": float(metrics_turkey["R2"]),
            "MAE": float(metrics_turkey["MAE"]),
            "RMSE": float(metrics_turkey["RMSE"])
        },
        "piecewise": {
            "break_year": float(best_break_year_turkey) if best_break_year_turkey is not None else None,
            "K": float(K_turkey),
            "beta1": float(params_turkey_piece.get("beta", 0)),
            "rho1": float(params_turkey_piece.get("rho", 0)),
            "beta2": float(params_turkey_piece.get("beta_post", params_turkey_piece.get("beta", 0))),
            "rho2": float(params_turkey_piece.get("rho_post", params_turkey_piece.get("rho", 0))),
            "R2": float(metrics_turkey_piece["R2"]),
            "MAE": float(metrics_turkey_piece["MAE"]),
            "RMSE": float(metrics_turkey_piece["RMSE"])
        }
    }
    turkey_summary_path = base_dir / "outputs" / "runs" / "jw_turkey_fit_summary.json"
    with open(turkey_summary_path, "w", encoding="utf-8") as f:
        json.dump(turkey_summary, f, indent=2)
    print(f"  Saved: {turkey_summary_path}")
    
    # Panel (c): New Zealand - baseline + piecewise
    print("\nLoading New Zealand data...")
    nz_path = base_dir / "data" / "nz" / "timeseries" / "nz_none_share_2001_2006_2013_2018_2023.csv"
    if not nz_path.exists():
        # Fallback to old file
        nz_path_old = base_dir / "data" / "nz" / "timeseries" / "nz_none_vs_religion_2013_2018_2023.csv"
        if nz_path_old.exists():
            nz_path = nz_path_old
        else:
            print(f"ERROR: New Zealand CSV not found. Tried:", file=sys.stderr)
            print(f"  - {base_dir / 'data' / 'nz' / 'timeseries' / 'nz_none_share_2001_2006_2013_2018_2023.csv'}", file=sys.stderr)
            print(f"  - {base_dir / 'data' / 'nz' / 'timeseries' / 'nz_none_vs_religion_2013_2018_2023.csv'}", file=sys.stderr)
            sys.exit(1)
    
    years_nz, y_obs_nz = load_time_series(nz_path, "year", "share_none")
    t0_nz = float(years_nz.min())
    t1_nz = float(years_nz.max())
    y0_nz = float(y_obs_nz[0])
    
    print(f"New Zealand: {len(years_nz)} points ({int(t0_nz)}-{int(t1_nz)})")
    
    # Fit New Zealand - baseline
    print("Fitting New Zealand baseline...")
    params_nz, metrics_nz, y_pred_nz, t_sim_nz, y_sim_nz = fit_baseline(
        years_nz, y_obs_nz, t0_nz, t1_nz, y0_nz
    )
    baseline_r2_nz = metrics_nz['R2']
    print(f"  Baseline R² = {baseline_r2_nz:.3f}")
    
    # Fit New Zealand - piecewise (beta constant, only rho changes)
    # Grid search for best break_year (2006, 2013, 2018)
    print("Fitting New Zealand piecewise (rho only, grid search for break_year)...")
    break_years_candidates_nz = [2006.0, 2013.0, 2018.0]
    best_r2_nz_piece = -np.inf
    best_break_year_nz = None
    best_params_nz_piece = None
    best_metrics_nz_piece = None
    best_y_pred_nz_piece = None
    best_t_sim_nz_piece = None
    best_y_sim_nz_piece = None
    
    for break_year_candidate in break_years_candidates_nz:
        if break_year_candidate <= t0_nz or break_year_candidate >= t1_nz:
            continue
        try:
            params_piece, metrics_piece, y_pred_piece, t_sim_piece, y_sim_piece = fit_nz_piecewise_rho(
                years_nz, y_obs_nz, t0_nz, t1_nz, y0_nz, break_year_candidate
            )
            print(f"  break_year={break_year_candidate:.1f}: R²={metrics_piece['R2']:.3f}, beta={params_piece.get('beta', 0):.4f}, rho_pre={params_piece.get('rho_pre', 0):.4f}, rho_post={params_piece.get('rho_post', 0):.4f}")
            if metrics_piece['R2'] > best_r2_nz_piece:
                best_r2_nz_piece = metrics_piece['R2']
                best_break_year_nz = break_year_candidate
                best_params_nz_piece = params_piece
                best_metrics_nz_piece = metrics_piece
                best_y_pred_nz_piece = y_pred_piece
                best_t_sim_nz_piece = t_sim_piece
                best_y_sim_nz_piece = y_sim_piece
        except Exception as e:
            print(f"  break_year={break_year_candidate:.1f}: failed ({e})")
            continue
    
    # Try anchored piecewise model (separate fits for pre/post regimes)
    # Use break_year=2013 (where the jump occurs)
    print("Fitting New Zealand anchored piecewise (separate pre/post fits, break_year=2013)...")
    best_r2_nz_anchored = -np.inf
    best_break_year_nz_anchored = 2013.0  # Fixed to 2013 where jump occurs
    best_params_nz_anchored = None
    best_metrics_nz_anchored = None
    best_y_pred_nz_anchored = None
    best_t_sim_nz_anchored = None
    best_y_sim_nz_anchored = None
    
    try:
        params_anchored, metrics_anchored, y_pred_anchored, t_sim_anchored, y_sim_anchored = fit_nz_anchored_piecewise(
            years_nz, y_obs_nz, best_break_year_nz_anchored
        )
        print(f"  Anchored break_year={best_break_year_nz_anchored:.1f}: R²={metrics_anchored['R2']:.3f}, beta_pre={params_anchored.get('beta_pre', 0):.4f}, rho_pre={params_anchored.get('rho_pre', 0):.4f}, beta_post={params_anchored.get('beta_post', 0):.4f}, rho_post={params_anchored.get('rho_post', 0):.4f}")
        best_r2_nz_anchored = metrics_anchored['R2']
        best_params_nz_anchored = params_anchored
        best_metrics_nz_anchored = metrics_anchored
        best_y_pred_nz_anchored = y_pred_anchored
        best_t_sim_nz_anchored = t_sim_anchored
        best_y_sim_nz_anchored = y_sim_anchored
    except Exception as e:
        print(f"  Anchored piecewise failed: {e}")
        best_break_year_nz_anchored = None
    
    # Choose best piecewise model (regular or anchored)
    best_r2_nz_piece_final = max(best_r2_nz_piece, best_r2_nz_anchored)
    use_anchored = best_r2_nz_anchored > best_r2_nz_piece
    
    if use_anchored and best_break_year_nz_anchored is not None:
        best_break_year_nz = best_break_year_nz_anchored
        best_params_nz_piece = best_params_nz_anchored
        best_metrics_nz_piece = best_metrics_nz_anchored
        best_y_pred_nz_piece = best_y_pred_nz_anchored
        best_t_sim_nz_piece = best_t_sim_nz_anchored
        best_y_sim_nz_piece = best_y_sim_nz_anchored
        print(f"  Using anchored model: break_year={best_break_year_nz:.1f}, R²={best_r2_nz_anchored:.3f}")
    
    # Only use piecewise if R² improves baseline by at least 0.01
    r2_improvement = best_r2_nz_piece_final - baseline_r2_nz if best_break_year_nz is not None else -np.inf
    if best_break_year_nz is not None and r2_improvement >= 0.01:
        params_nz_piece = best_params_nz_piece
        metrics_nz_piece = best_metrics_nz_piece
        y_pred_nz_piece = best_y_pred_nz_piece
        t_sim_nz_piece = best_t_sim_nz_piece
        y_sim_nz_piece = best_y_sim_nz_piece
        print(f"  Best piecewise: break_year={best_break_year_nz:.1f}, R²={best_r2_nz_piece_final:.3f} (improvement: +{r2_improvement:.3f})")
    else:
        # Use baseline only (piecewise doesn't improve enough)
        params_nz_piece = None
        metrics_nz_piece = None
        y_pred_nz_piece = None
        t_sim_nz_piece = None
        y_sim_nz_piece = None
        best_break_year_nz = None
        if best_break_year_nz is not None:
            print(f"  Piecewise R² = {best_r2_nz_piece_final:.3f} does not improve baseline ({baseline_r2_nz:.3f}) by >= 0.01, using baseline only")
        else:
            print(f"  No valid piecewise fit found, using baseline only")
    
    # Save New Zealand predictions and summary (for split figure script)
    print("Saving New Zealand fit results...")
    nz_pred_df = pd.DataFrame({
        "year": years_nz,
        "obs_share": y_obs_nz,
        "pred_baseline": y_pred_nz,
        "pred_piecewise": y_pred_nz_piece if y_pred_nz_piece is not None else y_pred_nz
    })
    nz_pred_path = base_dir / "outputs" / "runs" / "nz_predictions.csv"
    nz_pred_path.parent.mkdir(parents=True, exist_ok=True)
    nz_pred_df.to_csv(nz_pred_path, index=False)
    print(f"  Saved: {nz_pred_path}")
    
    nz_summary = {
        "dataset": "New Zealand: No religion share (2001-2023)",
        "baseline": {
            "beta": float(params_nz.get("beta", 0)),
            "rho": float(params_nz.get("rho", 0)),
            "R2": float(metrics_nz["R2"]),
            "MAE": float(metrics_nz["MAE"]),
            "RMSE": float(metrics_nz["RMSE"])
        },
        "piecewise": {
            "break_year": float(best_break_year_nz) if best_break_year_nz is not None else None,
            "beta": float(best_params_nz_piece.get("beta", params_nz_piece.get("beta_pre", 0) if params_nz_piece else 0)) if best_params_nz_piece else None,
            "beta_pre": float(best_params_nz_piece.get("beta_pre", 0)) if best_params_nz_piece else None,
            "beta_post": float(best_params_nz_piece.get("beta_post", 0)) if best_params_nz_piece else None,
            "rho_pre": float(best_params_nz_piece.get("rho_pre", 0)) if best_params_nz_piece else None,
            "rho_post": float(best_params_nz_piece.get("rho_post", 0)) if best_params_nz_piece else None,
            "R2": float(metrics_nz_piece["R2"]) if metrics_nz_piece else float(metrics_nz["R2"]),
            "MAE": float(metrics_nz_piece["MAE"]) if metrics_nz_piece else float(metrics_nz["MAE"]),
            "RMSE": float(metrics_nz_piece["RMSE"]) if metrics_nz_piece else float(metrics_nz["RMSE"])
        }
    }
    nz_summary_path = base_dir / "outputs" / "runs" / "nz_fit_summary.json"
    with open(nz_summary_path, "w", encoding="utf-8") as f:
        json.dump(nz_summary, f, indent=2)
    print(f"  Saved: {nz_summary_path}")
    
    # Create 3-panel figure
    print("\nCreating 3-panel figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    
    # Helper function to apply professional axis styling
    def apply_axis_style(ax, xmin, xmax, ymin, ymax, sci_y=False, y_decimals=3):
        """Apply professional axis styling with exactly 11 ticks."""
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # 11 x-tick (min-max)
        xt = np.linspace(xmin, xmax, 11)
        ax.set_xticks(xt)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v))}"))
        
        # 11 y-tick
        yt = np.linspace(ymin, ymax, 11)
        ax.set_yticks(yt)
        
        if sci_y:
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.{y_decimals}f}"))
        
        # Rotate x labels to avoid overlap
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha("right")
        
        ax.tick_params(labelsize=8)
    
    # Panel (a): Sweden
    ax = axes[0]
    ax.scatter(years_sweden, y_obs_sweden, s=25, label="Observed", color="black", alpha=0.7, zorder=3)
    ax.plot(t_sim_sweden_base, y_sim_sweden_base, linewidth=2, 
            label=f"Baseline ($R^2$={metrics_sweden_base['R2']:.3f})", zorder=2)
    ax.plot(t_sim_sweden_piece, y_sim_sweden_piece, linewidth=2, linestyle="--",
            label=f"Piecewise ($R^2$={metrics_sweden_piece['R2']:.3f})", zorder=2)
    ax.axvline(break_year_sweden, linestyle=":", linewidth=1, color="gray", alpha=0.5, zorder=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_title("(a) Sweden")
    # Set x and y limits based on data
    xmin_swe = float(years_sweden.min())
    xmax_swe = float(years_sweden.max())
    ymin_swe = min(float(y_obs_sweden.min()), float(y_sim_sweden_base.min()), float(y_sim_sweden_piece.min()))
    ymax_swe = max(float(y_obs_sweden.max()), float(y_sim_sweden_base.max()), float(y_sim_sweden_piece.max()))
    ymin_swe = max(0.0, ymin_swe * 0.95)
    ymax_swe = min(1.0, ymax_swe * 1.05)
    apply_axis_style(ax, xmin_swe, xmax_swe, ymin_swe, ymax_swe, sci_y=False, y_decimals=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Turkey
    ax = axes[1]
    ax.scatter(years_turkey, y_obs_turkey, s=30, label="Observed", color="black", alpha=0.7, zorder=3)
    ax.plot(t_sim_turkey, y_sim_turkey, linewidth=2, 
            label=f"Baseline ($R^2$={metrics_turkey['R2']:.3f})", zorder=2)
    if best_break_year_turkey is not None:
        ax.plot(t_sim_turkey_piece, y_sim_turkey_piece, linewidth=2, linestyle="--",
                label=f"Piecewise ($R^2$={metrics_turkey_piece['R2']:.3f})", zorder=2)
        ax.axvline(best_break_year_turkey, linestyle=":", linewidth=1, color="gray", alpha=0.5, zorder=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_title("(b) Turkey (JW)")
    # Set x and y limits based on data
    xmin_tur = float(years_turkey.min())
    xmax_tur = float(years_turkey.max())
    ymin_tur = min(float(y_obs_turkey.min()), float(y_sim_turkey.min())) * 0.95
    ymax_tur = max(float(y_obs_turkey.max()), float(y_sim_turkey.max())) * 1.05
    if best_break_year_turkey is not None:
        ymin_tur = min(ymin_tur, float(y_sim_turkey_piece.min()) * 0.95)
        ymax_tur = max(ymax_tur, float(y_sim_turkey_piece.max()) * 1.05)
    apply_axis_style(ax, xmin_tur, xmax_tur, ymin_tur, ymax_tur, sci_y=True, y_decimals=3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (c): New Zealand
    ax = axes[2]
    ax.scatter(years_nz, y_obs_nz, s=50, label="Observed", color="black", alpha=0.7, zorder=3)
    ax.plot(t_sim_nz, y_sim_nz, linewidth=2, 
            label=f"Baseline ($R^2$={metrics_nz['R2']:.3f})", zorder=2)
    if best_break_year_nz is not None and metrics_nz_piece is not None:
        ax.plot(t_sim_nz_piece, y_sim_nz_piece, linewidth=2, linestyle="--",
                label=f"Piecewise ($R^2$={metrics_nz_piece['R2']:.3f})", zorder=2)
        ax.axvline(best_break_year_nz, linestyle=":", linewidth=1, color="gray", alpha=0.5, zorder=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_title("(c) New Zealand")
    # Set x and y limits based on data
    xmin_nz = float(years_nz.min())
    xmax_nz = float(years_nz.max())
    ymin_nz = min(float(y_obs_nz.min()), float(y_sim_nz.min()))
    ymax_nz = max(float(y_obs_nz.max()), float(y_sim_nz.max()))
    if best_break_year_nz is not None and y_sim_nz_piece is not None:
        ymin_nz = min(ymin_nz, float(y_sim_nz_piece.min()))
        ymax_nz = max(ymax_nz, float(y_sim_nz_piece.max()))
    ymin_nz = max(0.0, ymin_nz * 0.95)
    ymax_nz = min(1.0, ymax_nz * 1.05)
    apply_axis_style(ax, xmin_nz, xmax_nz, ymin_nz, ymax_nz, sci_y=False, y_decimals=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Save figures
    out_figs = Path("outputs/figs")
    out_figs.mkdir(parents=True, exist_ok=True)
    png_path = out_figs / "fig_historic_fit.png"
    pdf_path = out_figs / "fig_historic_fit.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    
    print(f"\nSaved figures:")
    print(f"  {png_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
