"""
Proof-of-concept fitting script for Swedish church membership data.

Fits reduced ODE model with baseline and piecewise parameterizations.
Legacy script - superseded by histfit_reduced.py.
"""
import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_share_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to infer the membership share series from common column names.
    Expected output columns: year (int), share (float in [0,1]).
    """
    cols = {c.lower(): c for c in df.columns}

    # year
    if "year" in cols:
        year_col = cols["year"]
    elif "år" in cols:
        year_col = cols["år"]
    elif "ar" in cols:
        year_col = cols["ar"]
    else:
        raise ValueError("CSV must contain a 'year' column (or 'År' or 'Ar').")

    # share - look for percentage column
    share_col = None
    share = None
    
    # First, try to find percentage column by checking all columns
    for col in df.columns:
        col_lower = col.lower()
        if "%" in col or "percent" in col_lower or "pct" in col_lower or "folkmängden" in col_lower:
            share = df[col].astype(float).to_numpy()
            # if looks like percent (values > 1.5)
            if len(share[~np.isnan(share)]) > 0 and np.nanmax(share) > 1.5:
                share = share / 100.0
            share_col = col
            break
    
    # Fallback to standard column names
    if share is None:
        if "share" in cols:
            share_col = cols["share"]
            share = df[share_col].astype(float).to_numpy()
            if np.nanmax(share) > 1.5:
                share = share / 100.0
        elif "share_pct" in cols:
            share = df[cols["share_pct"]].astype(float).to_numpy() / 100.0
        elif "share_percent" in cols:
            share = df[cols["share_percent"]].astype(float).to_numpy() / 100.0
        elif "membership_share" in cols:
            share = df[cols["membership_share"]].astype(float).to_numpy()
            if np.nanmax(share) > 1.5:
                share = share / 100.0
        elif "members" in cols and "population" in cols:
            share = df[cols["members"]].astype(float).to_numpy() / df[cols["population"]].astype(float).to_numpy()
        else:
            raise ValueError(
                f"Could not infer share. Available columns: {list(df.columns)}. "
                "Provide a column with '%' or 'percent' in the name."
            )

    out = pd.DataFrame({
        "year": df[year_col].astype(int).to_numpy(),
        "share": share
    })
    out = out.dropna(subset=["share"]).sort_values("year").reset_index(drop=True)

    # Clamp just in case
    out["share"] = out["share"].clip(lower=0.0, upper=1.0)
    return out


# ----------------------------
# 1-strain ODE model (S, B, M, P)
# ----------------------------

@dataclass
class OneStrainParams:
    # demography
    b: float = 0.01
    mu: float = 0.01

    # conversion
    beta0: float = 0.3        # baseline conversion strength
    q: float = 0.05           # fraction entering as missionary

    # role transitions
    sigma: float = 0.02       # B -> M
    kappa: float = 0.02       # M -> B
    tauB: float = 0.002       # B -> P
    tauM: float = 0.003       # M -> P

    # disaffiliation
    rhoB: float = 0.006       # B -> S
    rhoM: float = 0.006       # M -> S
    rhoP: float = 0.001       # P -> S

    # piecewise context
    t_change: Optional[float] = None  # in model time units (years from start)
    beta_mult_post: float = 1.0       # multiplier after change


def ode_rhs(state: np.ndarray, p: OneStrainParams, t: float) -> np.ndarray:
    """
    state = [S, B, M, P] counts (or proportions; equations are scale-consistent).
    """
    S, B, M, P = state
    N = S + B + M + P
    if N <= 0:
        return np.zeros_like(state)

    # piecewise beta(t)
    beta = p.beta0
    if p.t_change is not None and t >= p.t_change:
        beta = p.beta0 * p.beta_mult_post

    lam = beta * (M / N)  # lambda(t) = beta(t) * M/N

    # For 1 strain: X = S (no cross conversion from other religions)
    X = S

    dS = p.b * S - p.mu * S - lam * S + (p.rhoB * B + p.rhoM * M + p.rhoP * P)

    dB = p.b * B - p.mu * B
    dB += (1.0 - p.q) * lam * X
    dB += -p.sigma * B + p.kappa * M
    dB += -p.rhoB * B - p.tauB * B

    dM = p.b * M - p.mu * M
    dM += p.q * lam * X
    dM += p.sigma * B - p.kappa * M
    dM += -p.rhoM * M - p.tauM * M

    dP = p.b * P - p.mu * P
    dP += p.tauB * B + p.tauM * M
    dP += -p.rhoP * P

    return np.array([dS, dB, dM, dP], dtype=float)


def rk4_integrate(
    y0: np.ndarray,
    p: OneStrainParams,
    t0: float,
    t1: float,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 integration. Returns (times, states).
    """
    n_steps = int(math.ceil((t1 - t0) / dt))
    times = np.linspace(t0, t1, n_steps + 1)
    states = np.zeros((n_steps + 1, len(y0)), dtype=float)
    states[0] = y0.astype(float)

    y = y0.astype(float)
    t = t0
    for i in range(n_steps):
        h = times[i + 1] - times[i]
        k1 = ode_rhs(y, p, t)
        k2 = ode_rhs(y + 0.5 * h * k1, p, t + 0.5 * h)
        k3 = ode_rhs(y + 0.5 * h * k2, p, t + 0.5 * h)
        k4 = ode_rhs(y + h * k3, p, t + h)
        y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # keep nonnegative
        y = np.maximum(y, 0.0)

        states[i + 1] = y
        t = times[i + 1]

    return times, states


def simulate_share_over_years(
    years: np.ndarray,
    share0: float,
    p: OneStrainParams,
    dt: float = 0.1,
    init_M_frac: float = 0.02,
    init_P_frac: float = 0.00
) -> np.ndarray:
    """
    Simulate and return y(t) = (B+M+P)/N evaluated at integer-year grid.
    Uses counts with N=1.0 (proportions) for simplicity.
    """
    years = np.asarray(years, dtype=int)
    y_start = years.min()
    t_end = float(years.max() - y_start)

    # initial composition: total church share = share0
    church0 = float(share0)
    church0 = min(max(church0, 0.0), 1.0)

    M0 = church0 * init_M_frac
    P0 = church0 * init_P_frac
    B0 = max(church0 - M0 - P0, 0.0)
    S0 = 1.0 - (B0 + M0 + P0)

    y0 = np.array([S0, B0, M0, P0], dtype=float)

    times, states = rk4_integrate(y0, p, t0=0.0, t1=t_end, dt=dt)

    # evaluate at each year offset
    preds = []
    for yr in years:
        t = float(yr - y_start)
        idx = int(np.argmin(np.abs(times - t)))
        S, B, M, P = states[idx]
        N = S + B + M + P
        preds.append((B + M + P) / N if N > 0 else 0.0)

    return np.array(preds, dtype=float)


# ----------------------------
# Fitting (simple random search)
# ----------------------------

def fit_baseline(
    years: np.ndarray,
    obs_share: np.ndarray,
    t_change: Optional[float],
    n_samples: int,
    seed: int = 42
) -> Tuple[OneStrainParams, Dict[str, float], np.ndarray]:
    rng = np.random.default_rng(seed)

    # Fixed "nuisance" params (keep minimal for identifiability)
    base = OneStrainParams(
        b=0.01, mu=0.01,
        q=0.05, sigma=0.02, kappa=0.02,
        tauB=0.002, tauM=0.003,
        rhoP=0.001,
        t_change=t_change,
        beta_mult_post=1.0
    )

    best = None
    best_loss = float("inf")
    best_pred = None

    # search ranges (log-uniform where reasonable)
    for _ in range(n_samples):
        beta0 = float(10 ** rng.uniform(np.log10(0.05), np.log10(1.0)))
        rho = float(10 ** rng.uniform(np.log10(1e-4), np.log10(5e-2)))

        p = OneStrainParams(
            b=base.b, mu=base.mu,
            beta0=beta0,
            q=base.q, sigma=base.sigma, kappa=base.kappa,
            tauB=base.tauB, tauM=base.tauM,
            rhoB=rho, rhoM=rho, rhoP=base.rhoP,
            t_change=base.t_change,
            beta_mult_post=base.beta_mult_post
        )

        pred = simulate_share_over_years(years, obs_share[0], p, dt=0.1)

        # loss = SSE
        loss = float(np.mean((pred - obs_share) ** 2))
        if loss < best_loss:
            best_loss = loss
            best = p
            best_pred = pred

    metrics = {
        "MAE": mae(obs_share, best_pred),
        "RMSE": rmse(obs_share, best_pred),
        "R2": r2_score(obs_share, best_pred),
        "MSE": float(best_loss)
    }
    return best, metrics, best_pred


def fit_piecewise_beta(
    years: np.ndarray,
    obs_share: np.ndarray,
    t_change: float,
    n_samples: int,
    seed: int = 43
) -> Tuple[OneStrainParams, Dict[str, float], np.ndarray]:
    rng = np.random.default_rng(seed)

    base = OneStrainParams(
        b=0.01, mu=0.01,
        q=0.05, sigma=0.02, kappa=0.02,
        tauB=0.002, tauM=0.003,
        rhoP=0.001,
        t_change=t_change,
        beta_mult_post=1.0
    )

    best = None
    best_loss = float("inf")
    best_pred = None

    for _ in range(n_samples):
        beta0 = float(10 ** rng.uniform(np.log10(0.05), np.log10(1.0)))
        rho = float(10 ** rng.uniform(np.log10(1e-4), np.log10(5e-2)))
        # post-change multiplier: secularization -> typically <= 1
        m = float(10 ** rng.uniform(np.log10(0.1), np.log10(1.0)))

        p = OneStrainParams(
            b=base.b, mu=base.mu,
            beta0=beta0,
            q=base.q, sigma=base.sigma, kappa=base.kappa,
            tauB=base.tauB, tauM=base.tauM,
            rhoB=rho, rhoM=rho, rhoP=base.rhoP,
            t_change=t_change,
            beta_mult_post=m
        )

        pred = simulate_share_over_years(years, obs_share[0], p, dt=0.1)

        loss = float(np.mean((pred - obs_share) ** 2))
        if loss < best_loss:
            best_loss = loss
            best = p
            best_pred = pred

    metrics = {
        "MAE": mae(obs_share, best_pred),
        "RMSE": rmse(obs_share, best_pred),
        "R2": r2_score(obs_share, best_pred),
        "MSE": float(best_loss)
    }
    return best, metrics, best_pred


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="outputs/runs/swedish_church_membership_1972_2024.csv")
    ap.add_argument("--change_year", type=int, default=2000, help="institutional change year for piecewise context")
    ap.add_argument("--baseline_samples", type=int, default=3000)
    ap.add_argument("--piecewise_samples", type=int, default=6000)
    ap.add_argument("--out_prefix", type=str, default="outputs/runs/sweden_poc")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    data = safe_share_from_csv(df)

    years = data["year"].to_numpy(dtype=int)
    obs = data["share"].to_numpy(dtype=float)

    # Drop any NaNs just in case
    mask = np.isfinite(obs)
    years = years[mask]
    obs = obs[mask]

    start_year = int(years.min())
    t_change = float(args.change_year - start_year)
    if t_change <= 0:
        raise ValueError(f"change_year must be after start_year={start_year}")

    print("=" * 60)
    print(f"Fitting Swedish Church membership data ({start_year}-{years.max()})")
    print(f"Change year: {args.change_year} (t={t_change:.1f} years from start)")
    print("=" * 60)

    # Fit baseline (constant beta)
    print("\nFitting baseline model (constant beta)...")
    p_base, m_base, pred_base = fit_baseline(
        years=years,
        obs_share=obs,
        t_change=None,
        n_samples=args.baseline_samples
    )
    print(f"Baseline fit complete. R² = {m_base['R2']:.4f}")

    # Fit piecewise beta
    print(f"\nFitting piecewise model (beta change at {args.change_year})...")
    p_piece, m_piece, pred_piece = fit_piecewise_beta(
        years=years,
        obs_share=obs,
        t_change=t_change,
        n_samples=args.piecewise_samples
    )
    print(f"Piecewise fit complete. R² = {m_piece['R2']:.4f}")

    # Save predictions CSV
    out_pred = pd.DataFrame({
        "year": years,
        "obs_share": obs,
        "pred_baseline": pred_base,
        "pred_piecewise": pred_piece
    })
    pred_path = f"{args.out_prefix}_predictions.csv"
    out_pred.to_csv(pred_path, index=False)
    print(f"\nSaved predictions: {pred_path}")

    # Save summary JSON
    summary = {
        "data": {"csv": args.csv, "start_year": start_year, "end_year": int(years.max())},
        "baseline": {"params": vars(p_base), "metrics": m_base},
        "piecewise_beta": {"params": vars(p_piece), "metrics": m_piece},
        "notes": {
            "model": "1-strain ODE with compartments S,B,M,P; fitted to y(t)=(B+M+P)/N",
            "piecewise": f"beta(t)=beta0 pre-{args.change_year}, beta0*beta_mult_post post-{args.change_year}"
        }
    }
    summary_path = f"{args.out_prefix}_fit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Plot (no title; paper-friendly)
    fig_path_pdf = f"outputs/figures/fig_sweden_poc_fit.pdf"
    fig_path_png = f"outputs/figures/fig_sweden_poc_fit.png"

    plt.figure(figsize=(8, 6))
    plt.plot(years, obs, marker="o", linestyle="None", markersize=4, label="Observed (membership share)", color="black", alpha=0.7)
    plt.plot(years, pred_base, label=f"Model (baseline, R²={m_base['R2']:.3f})", linewidth=2)
    plt.plot(years, pred_piece, label=f"Model (piecewise, R²={m_piece['R2']:.3f})", linewidth=2)

    plt.xlabel("Year")
    plt.ylabel("Share")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    from pathlib import Path
    Path(fig_path_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path_pdf)
    plt.savefig(fig_path_png, dpi=200)
    print(f"Saved figures: {fig_path_pdf}, {fig_path_png}")

    print("\n" + "=" * 60)
    print("=== Results ===")
    print("Baseline metrics:", m_base)
    print("Piecewise metrics:", m_piece)
    print("=" * 60)


if __name__ == "__main__":
    main()

