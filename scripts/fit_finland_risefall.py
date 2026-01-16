#!/usr/bin/env python3
"""
Fit Finland OTHER RELIGIOUS GROUPS (rise-and-fall) time series.
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

DATA = Path("data/raw/finland/finland_risefall_otherreliggroups_1990_2019.csv")
OUTDIR = Path("outputs/runs")
OUTDIR.mkdir(parents=True, exist_ok=True)

def rhs(t, y, beta, rho):
    return beta * y * (1 - y) - rho * y

def sim_baseline(years, y0, beta, rho):
    years = np.asarray(years, float)
    t = years - years[0]
    sol = solve_ivp(lambda tt, yy: rhs(tt, yy, beta, rho),
                    (t[0], t[-1]), [y0], t_eval=t, rtol=1e-8, atol=1e-10)
    return sol.y[0]

def sim_piecewise(years, y0, b1, r1, b2, r2, break_year):
    years = np.asarray(years, float)
    t = years - years[0]
    tb = break_year - years[0]

    m1 = t <= tb
    t1 = t[m1]
    if len(t1) > 0:
        sol1 = solve_ivp(lambda tt, yy: rhs(tt, yy, b1, r1),
                         (t1[0], t1[-1]), [y0], t_eval=t1, rtol=1e-8, atol=1e-10)
        y1 = sol1.y[0]
        y_tb = y1[-1]
    else:
        y1 = np.array([], dtype=float)
        y_tb = y0

    m2 = t >= tb
    t2 = t[m2]
    if len(t2) > 0:
        sol2 = solve_ivp(lambda tt, yy: rhs(tt, yy, b2, r2),
                         (t2[0], t2[-1]), [y_tb], t_eval=t2, rtol=1e-8, atol=1e-10)
        y2 = sol2.y[0]
    else:
        y2 = np.array([], dtype=float)

    yhat = np.empty_like(t)
    yhat[m1] = y1
    yhat[m2] = y2
    return yhat

def metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def fit_baseline(years, y):
    y0 = float(y[0])
    def loss(th):
        b, r = th
        if b < -10 or b > 10 or r < -10 or r > 10:
            return 1e9
        yhat = sim_baseline(years, y0, b, r)
        return float(np.mean((y - yhat)**2))
    res = minimize(loss, x0=np.array([0.1, 0.1]), method="Nelder-Mead")
    b, r = map(float, res.x)
    return b, r, sim_baseline(years, y0, b, r)

def fit_piecewise(years, y):
    y0 = float(y[0])
    candidates = range(int(years[0]) + 3, int(years[-1]) - 2)
    best = None
    for by in candidates:
        def loss(th):
            b1, r1, b2, r2 = th
            if any([b1 < -10, b1 > 10, r1 < -10, r1 > 10, b2 < -10, b2 > 10, r2 < -10, r2 > 10]):
                return 1e9
            yhat = sim_piecewise(years, y0, b1, r1, b2, r2, by)
            return float(np.mean((y - yhat)**2))
        res = minimize(loss, x0=np.array([0.1, 0.1, 0.05, 0.2]), method="Nelder-Mead")
        b1, r1, b2, r2 = map(float, res.x)
        yhat = sim_piecewise(years, y0, b1, r1, b2, r2, by)
        cand = {"break_year": int(by), "b1": b1, "r1": r1, "b2": b2, "r2": r2,
                "mse": float(np.mean((y - yhat)**2)), "yhat": yhat}
        if best is None or cand["mse"] < best["mse"]:
            best = cand
    return best

def main():
    if not DATA.exists():
        print(f"ERROR: Data file not found: {DATA}", file=sys.stderr)
        print(f"  Run first: python scripts/build_finland_risefall_timeseries.py", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(DATA).sort_values("year")
    years = df["year"].to_numpy(dtype=int)
    y = df["obs_share"].to_numpy(dtype=float)

    print(f"Fitting Finland OTHER RELIGIOUS GROUPS ({len(years)} points, {years.min()}-{years.max()})")
    print(f"  Share range: {y.min():.6f} - {y.max():.6f}")

    print("Fitting baseline model...")
    b, r, yhat_b = fit_baseline(years, y)
    mb = metrics(y, yhat_b)
    print(f"  Baseline: beta={b:.6f}, rho={r:.6f}, R²={mb['R2']:.4f}")

    print("Fitting piecewise model (grid search for break_year)...")
    best = fit_piecewise(years, y)
    if best is None:
        print("ERROR: No valid piecewise fit found", file=sys.stderr)
        sys.exit(1)
    
    mp = metrics(y, best["yhat"])
    print(f"  Piecewise: break_year={best['break_year']}, R²={mp['R2']:.4f}")

    pred = pd.DataFrame({"year": years, "obs_share": y,
                         "pred_baseline": yhat_b, "pred_piecewise": best["yhat"]})
    pred_path = OUTDIR / "finland_risefall_predictions.csv"
    ensure_dirs(pred_path.parent)
    pred.to_csv(pred_path, index=False)

    summ = {
        "dataset": "Finland (OTHER RELIGIOUS GROUPS share), 1990-2019",
        "baseline": {"beta": b, "rho": r, **mb},
        "piecewise": {"break_year": best["break_year"],
                      "beta1": best["b1"], "rho1": best["r1"],
                      "beta2": best["b2"], "rho2": best["r2"],
                      **mp}
    }
    sum_path = OUTDIR / "finland_risefall_fit_summary.json"
    ensure_dirs(sum_path.parent)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    print(f"\nWrote: {pred_path}")
    print(f"Wrote: {sum_path}")
    print(f"\nBaseline: R²={mb['R2']:.4f}, MAE={mb['MAE']:.4f}")
    print(f"Piecewise: R²={mp['R2']:.4f}, MAE={mp['MAE']:.4f}, break_year={best['break_year']}")

if __name__ == "__main__":
    main()
