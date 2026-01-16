#!/usr/bin/env python3
"""
Fit New Zealand "none" group using reduced ODE model (baseline + piecewise).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dirs

DATA = Path("data/processed/nz_3groups_shares_long.csv")
OUT = Path("out")
OUT.mkdir(exist_ok=True, parents=True)

# Reduced calibration model (1D):
# dy/dt = beta * y*(1-y) - rho*y
def rhs(t, y, beta, rho):
    return beta * y * (1 - y) - rho * y

def simulate_baseline(years, y0, beta, rho):
    years = np.asarray(years, float)
    t = years - years[0]              # t=0..T
    sol = solve_ivp(lambda tt, yy: rhs(tt, yy, beta, rho),
                    (t[0], t[-1]), [y0], t_eval=t,
                    rtol=1e-8, atol=1e-10)
    return sol.y[0]

def simulate_piecewise(years, y0, beta1, rho1, beta2, rho2, break_year):
    years = np.asarray(years, float)
    t = years - years[0]
    tb = break_year - years[0]

    # segment 1: up to tb
    mask1 = t <= tb
    t1 = t[mask1]
    if len(t1) == 0:
        y_tb = y0
        y1 = np.array([], dtype=float)
    else:
        sol1 = solve_ivp(lambda tt, yy: rhs(tt, yy, beta1, rho1),
                         (t1[0], t1[-1]), [y0], t_eval=t1,
                         rtol=1e-8, atol=1e-10)
        y1 = sol1.y[0]
        y_tb = y1[-1]

    # segment 2: from tb
    mask2 = t >= tb
    t2 = t[mask2]
    if len(t2) == 0:
        y2 = np.array([], dtype=float)
    else:
        sol2 = solve_ivp(lambda tt, yy: rhs(tt, yy, beta2, rho2),
                         (t2[0], t2[-1]), [y_tb], t_eval=t2,
                         rtol=1e-8, atol=1e-10)
        y2 = sol2.y[0]

    # stitch back
    y_pred = np.empty_like(t)
    y_pred[mask1] = y1
    y_pred[mask2] = y2
    return y_pred

def calc_metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def fit_baseline(years, y_obs):
    y0 = float(y_obs[0])

    def loss(theta):
        beta, rho = theta
        if beta < -10 or beta > 10 or rho < -10 or rho > 10:
            return 1e9
        yhat = simulate_baseline(years, y0, beta, rho)
        return float(np.mean((y_obs - yhat) ** 2))

    res = minimize(loss, x0=np.array([0.1, 0.05]), method="Nelder-Mead")
    beta, rho = map(float, res.x)
    yhat = simulate_baseline(years, y0, beta, rho)
    return beta, rho, yhat

def fit_piecewise(years, y_obs, break_year_candidates):
    y0 = float(y_obs[0])
    best = None

    for by in break_year_candidates:
        # ignore too-close-to-edge breaks
        if by <= years[0] + 1 or by >= years[-1] - 1:
            continue

        def loss(theta):
            beta1, rho1, beta2, rho2 = theta
            if any([beta1 < -10, beta1 > 10, rho1 < -10, rho1 > 10,
                    beta2 < -10, beta2 > 10, rho2 < -10, rho2 > 10]):
                return 1e9
            yhat = simulate_piecewise(years, y0, beta1, rho1, beta2, rho2, by)
            return float(np.mean((y_obs - yhat) ** 2))

        res = minimize(loss, x0=np.array([0.1, 0.05, 0.05, 0.05]), method="Nelder-Mead")
        beta1, rho1, beta2, rho2 = map(float, res.x)
        yhat = simulate_piecewise(years, y0, beta1, rho1, beta2, rho2, by)

        cand = {
            "break_year": int(by),
            "beta1": beta1, "rho1": rho1,
            "beta2": beta2, "rho2": rho2,
            "yhat": yhat,
            "mse": float(np.mean((y_obs - yhat) ** 2))
        }
        if best is None or cand["mse"] < best["mse"]:
            best = cand

    return best

def main():
    if not DATA.exists():
        print(f"ERROR: Data file not found: {DATA}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(DATA)
    df = df[df["group"] == "none"].sort_values("year")
    
    if len(df) == 0:
        print(f"ERROR: No data found for group 'none'", file=sys.stderr)
        print(f"Available groups: {df['group'].unique() if 'group' in df.columns else 'N/A'}", file=sys.stderr)
        sys.exit(1)
    
    years = df["year"].to_numpy(dtype=int)
    y_obs = df["share"].to_numpy(dtype=float)

    print(f"Fitting New Zealand 'none' group ({len(years)} points, {years.min()}-{years.max()})")
    print(f"  Share range: {y_obs.min():.6f} - {y_obs.max():.6f}")

    # baseline
    print("\nFitting baseline model...")
    beta, rho, yhat_base = fit_baseline(years, y_obs)
    m_base = calc_metrics(y_obs, yhat_base)
    print(f"  Baseline: beta={beta:.6f}, rho={rho:.6f}, R²={m_base['R2']:.4f}")

    # piecewise (grid-search: 2006, 2013, 2018)
    print("\nFitting piecewise model (grid search for break_year in {2006, 2013, 2018})...")
    by_candidates = [2006, 2013, 2018]
    best = fit_piecewise(years, y_obs, by_candidates)
    
    if best is None:
        print("ERROR: No valid piecewise fit found", file=sys.stderr)
        sys.exit(1)
    
    m_piece = calc_metrics(y_obs, best["yhat"])
    print(f"  Piecewise: break_year={best['break_year']}, R²={m_piece['R2']:.4f}")
    print(f"    Pre-break: beta1={best['beta1']:.6f}, rho1={best['rho1']:.6f}")
    print(f"    Post-break: beta2={best['beta2']:.6f}, rho2={best['rho2']:.6f}")

    # save predictions
    pred = pd.DataFrame({
        "year": years,
        "obs_share": y_obs,
        "pred_baseline": yhat_base,
        "pred_piecewise": best["yhat"],
    })
    pred_path = OUT / "nz_none_predictions.csv"
    ensure_dirs(pred_path.parent)
    pred.to_csv(pred_path, index=False)

    # save summary
    summary = {
        "dataset": "New Zealand: No religion share (2001-2023)",
        "group": "none",
        "baseline": {"beta": beta, "rho": rho, **m_base},
        "piecewise": {
            "break_year": best["break_year"],
            "beta1": best["beta1"], "rho1": best["rho1"],
            "beta2": best["beta2"], "rho2": best["rho2"],
            **m_piece
        }
    }
    sum_path = OUT / "nz_none_fit_summary.json"
    ensure_dirs(sum_path.parent)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {pred_path}")
    print(f"Wrote: {sum_path}")
    print(f"\nBaseline: R²={m_base['R2']:.4f}, MAE={m_base['MAE']:.4f}")
    print(f"Piecewise: R²={m_piece['R2']:.4f}, MAE={m_piece['MAE']:.4f}, break_year={best['break_year']}")

if __name__ == "__main__":
    main()
