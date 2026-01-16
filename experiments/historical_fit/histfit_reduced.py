#!/usr/bin/env python3
"""
Unified historical fitting pipeline for reduced ODE model.

Fits the reduced ODE: dy/dt = beta(t)*y*(1-y) - rho(t)*y
- baseline mode: beta, rho are constant
- piecewise mode: beta, rho differ pre/post breakpoint

Usage:
    python experiments/histfit_reduced.py --csv data/sweden/file.csv --year_col Ar --share_col "Medlemmar % av folkmängden" --mode baseline --name sweden
    python experiments/histfit_reduced.py --csv data/sweden/file.csv --year_col Ar --share_col "Medlemmar % av folkmängden" --mode piecewise --break_year 2000 --name sweden
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def load_time_series(csv_path: Path, year_col: str, share_col: str) -> Tuple[np.ndarray, np.ndarray]:
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
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate reduced ODE: dy/dt = beta(t)*y*(1-y) - rho(t)*y
    
    Args:
        t0, t1: time range
        y0: initial condition
        beta, rho: baseline parameters (or pre-break if piecewise)
        beta_post, rho_post: post-break parameters (if piecewise)
        break_year: breakpoint year (if piecewise)
        dt: time step for simulation
    
    Returns:
        t_eval: time points
        y: solution array
    """
    def rhs(t, y):
        if break_year is not None and t >= break_year:
            beta_t = beta_post
            rho_t = rho_post
        else:
            beta_t = beta
            rho_t = rho
        return [beta_t * y[0] * (1.0 - y[0]) - rho_t * y[0]]
    
    t_eval = np.arange(t0, t1 + dt, dt)
    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0=[y0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    
    y = np.clip(sol.y[0], 0.0, 1.0)
    return sol.t, y


def compute_metrics(y_obs: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, MAE, RMSE metrics."""
    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    mae = float(np.mean(np.abs(y_obs - y_pred)))
    rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MSE": float(ss_res / len(y_obs))
    }


def fit_baseline(
    years: np.ndarray,
    y_obs: np.ndarray,
    t0: float,
    t1: float,
    y0: float
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Fit baseline model (constant beta, rho)."""
    def objective(x):
        beta, rho = x
        if beta <= 0 or rho <= 0:
            return 1e9
        try:
            t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho)
            y_pred = np.interp(years, t_sim, y_sim)
            loss = np.mean((y_obs - y_pred) ** 2)
            return loss
        except Exception:
            return 1e9
    
    x0 = np.array([0.15, 0.05])
    bounds = [(1e-6, 5.0), (1e-6, 5.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    
    beta, rho = res.x
    t_sim, y_sim = simulate_reduced_ode(t0, t1, y0, beta, rho)
    y_pred = np.interp(years, t_sim, y_sim)
    
    params = {
        "beta": float(beta),
        "rho": float(rho)
    }
    metrics = compute_metrics(y_obs, y_pred)
    
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
                break_year=break_year
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
        break_year=break_year
    )
    y_pred = np.interp(years, t_sim, y_sim)
    
    params = {
        "beta_pre": float(beta1),
        "beta_post": float(beta2),
        "rho_pre": float(rho1),
        "rho_post": float(rho2)
    }
    metrics = compute_metrics(y_obs, y_pred)
    
    return params, metrics, y_pred, t_sim, y_sim


def main():
    parser = argparse.ArgumentParser(description="Fit reduced ODE model to historical data")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--year_col", type=str, required=True, help="Year column name")
    parser.add_argument("--share_col", type=str, required=True, help="Share column name")
    parser.add_argument("--mode", type=str, choices=["baseline", "piecewise"], required=True, help="Fitting mode")
    parser.add_argument("--break_year", type=float, help="Break year for piecewise mode")
    parser.add_argument("--name", type=str, required=True, help="Output name prefix")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for simulation")
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    years, y_obs = load_time_series(csv_path, args.year_col, args.share_col)
    
    if len(years) == 0:
        print("ERROR: No valid data points found", file=sys.stderr)
        sys.exit(1)
    
    t0 = float(years.min())
    t1 = float(years.max())
    y0 = float(y_obs[0])
    
    print(f"Loaded {len(years)} data points ({int(t0)}-{int(t1)})")
    print(f"Initial share: {y0:.6f}, Final share: {y_obs[-1]:.6f}")
    
    # Fit model
    if args.mode == "baseline":
        params, metrics, y_pred, t_sim, y_sim = fit_baseline(years, y_obs, t0, t1, y0)
        break_year = None
    elif args.mode == "piecewise":
        if args.break_year is None:
            print("ERROR: --break_year required for piecewise mode", file=sys.stderr)
            sys.exit(1)
        params, metrics, y_pred, t_sim, y_sim = fit_piecewise(
            years, y_obs, t0, t1, y0, float(args.break_year)
        )
        break_year = float(args.break_year)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print(f"\nFitting complete. R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.6f}")
    
    # Save summary JSON
    summary = {
        "data": {
            "csv": str(csv_path),
            "year_col": args.year_col,
            "share_col": args.share_col,
            "start_year": int(t0),
            "end_year": int(t1),
            "n_points": len(years)
        },
        "mode": args.mode,
        "break_year": int(break_year) if break_year is not None else None,
        "params": params,
        "metrics": metrics,
        "model": "dy/dt = beta(t)*y*(1-y) - rho(t)*y"
    }
    
    out_runs = Path("outputs/runs")
    out_runs.mkdir(parents=True, exist_ok=True)
    json_path = out_runs / f"{args.name}_{args.mode}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")
    
    # Save predictions CSV
    pred_df = pd.DataFrame({
        "year": years,
        "y_obs": y_obs,
        "y_hat": y_pred
    })
    csv_pred_path = out_runs / f"{args.name}_{args.mode}_predictions.csv"
    pred_df.to_csv(csv_pred_path, index=False)
    print(f"Saved predictions: {csv_pred_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(years, y_obs, s=30, label="Observed", color="black", alpha=0.7, zorder=3)
    ax.plot(t_sim, y_sim, linewidth=2, label=f"Model ($R^2$={metrics['R2']:.3f})", zorder=2)
    
    if break_year is not None:
        ax.axvline(break_year, linestyle="--", linewidth=1, color="gray", alpha=0.7, label=f"Break = {int(break_year)}")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_figs = Path("outputs/figs")
    out_figs.mkdir(parents=True, exist_ok=True)
    png_path = out_figs / f"{args.name}_{args.mode}_fit.png"
    pdf_path = out_figs / f"{args.name}_{args.mode}_fit.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved figures: {png_path}, {pdf_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
