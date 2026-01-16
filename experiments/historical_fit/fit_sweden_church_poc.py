"""
Proof-of-concept fitting for Swedish church membership decline.

Fits reduced ODE with baseline and piecewise beta/rho parameters.
Legacy script - superseded by histfit_reduced.py.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "outputs/runs/swedish_church_membership_1972_2024.csv"
OUT_PNG = "outputs/figures/fig_sweden_poc_fit.png"
OUT_PDF = "outputs/figures/fig_sweden_poc_fit.pdf"
OUT_JSON = "outputs/runs/sweden_poc_fit_summary.json"
OUT_CSV = "outputs/runs/sweden_poc_predictions.csv"

BREAK_YEAR = 2000.0
DT = 0.1  # years

# -----------------------------
# Data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Find year column
year_col = None
for col in df.columns:
    col_lower = col.lower()
    if "year" in col_lower or "år" in col_lower or "ar" in col_lower:
        year_col = col
        break

if year_col is None:
    raise ValueError(f"CSV must contain 'year' column. Columns: {list(df.columns)}")

# Find share/percentage column
share_col = None
for col in df.columns:
    col_lower = col.lower()
    if "%" in col or "percent" in col_lower or "pct" in col_lower or "folkmängden" in col_lower:
        share_col = col
        break

if share_col is None:
    # Try to use last numeric column
    for col in reversed(df.columns):
        if col != year_col and df[col].dtype in [np.float64, np.int64]:
            share_col = col
            break

if share_col is None:
    raise ValueError(f"Could not find share column. Columns: {list(df.columns)}")

years = df[year_col].to_numpy(dtype=float)
y_obs = df[share_col].to_numpy(dtype=float)

# Convert percentage to share if needed
if np.nanmax(y_obs) > 1.5:
    y_obs = y_obs / 100.0

# Normalize guard and drop NaN
mask = np.isfinite(y_obs) & np.isfinite(years)
years = years[mask]
y_obs = y_obs[mask]
y_obs = np.clip(y_obs, 0.0, 1.0)

t0 = years.min()
t1 = years.max()
y0 = float(y_obs[0])

print(f"Data loaded: {len(years)} years ({int(t0)}-{int(t1)})")
print(f"Initial share: {y0:.3f}, Final share: {y_obs[-1]:.3f}")

# -----------------------------
# Model (1-strain PoC)
# dy/dt = beta(t) * y*(1-y)  - rho(t)*y
# -----------------------------
def simulate(beta1, beta2, rho1, rho2, break_year=BREAK_YEAR):
    def rhs(t, y):
        beta = beta1 if t < break_year else beta2
        rho = rho1 if t < break_year else rho2
        # y is array([val])
        return [beta * y[0] * (1.0 - y[0]) - rho * y[0]]

    t_eval = np.arange(t0, t1 + 1e-9, DT)
    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0=[y0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    y_hat = np.interp(years, sol.t, sol.y[0])
    y_hat = np.clip(y_hat, 0.0, 1.0)
    return y_hat, sol.t, sol.y[0]

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# -----------------------------
# Fit baseline: beta1=beta2, rho1=rho2
# -----------------------------
def fit_baseline():
    def obj(x):
        beta, rho = x
        y_hat, *_ = simulate(beta, beta, rho, rho)
        return np.mean((y_obs - y_hat) ** 2)

    x0 = np.array([0.15, 0.05])
    bounds = [(1e-6, 5.0), (1e-6, 5.0)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    beta, rho = res.x
    y_hat, tt, yy = simulate(beta, beta, rho, rho)
    return (beta, rho), y_hat, (tt, yy)

# -----------------------------
# Fit piecewise: beta1,beta2,rho1,rho2
# -----------------------------
def fit_piecewise():
    def obj(x):
        beta1, beta2, rho1, rho2 = x
        # slight adjustment: prevent extreme values in post period
        if beta1 <= 0 or beta2 <= 0 or rho1 <= 0 or rho2 <= 0:
            return 1e9
        y_hat, *_ = simulate(beta1, beta2, rho1, rho2)
        return np.mean((y_obs - y_hat) ** 2)

    x0 = np.array([0.2, 0.1, 0.03, 0.08])
    bounds = [(1e-6, 5.0), (1e-6, 5.0), (1e-6, 5.0), (1e-6, 5.0)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    beta1, beta2, rho1, rho2 = res.x
    y_hat, tt, yy = simulate(beta1, beta2, rho1, rho2)
    return (beta1, beta2, rho1, rho2), y_hat, (tt, yy)

# -----------------------------
# Run fits
# -----------------------------
print("\n" + "=" * 60)
print("Fitting models...")
print("=" * 60)

print("\nFitting baseline model (constant beta and rho)...")
(b_beta, b_rho), y_base, (t_base, y_base_dense) = fit_baseline()

print("\nFitting piecewise model (beta and rho change at 2000)...")
(p_beta1, p_beta2, p_rho1, p_rho2), y_piece, (t_piece, y_piece_dense) = fit_piecewise()

r2_base = r2_score(y_obs, y_base)
r2_piece = r2_score(y_obs, y_piece)
mae_base = mae(y_obs, y_base)
mae_piece = mae(y_obs, y_piece)
rmse_base = rmse(y_obs, y_base)
rmse_piece = rmse(y_obs, y_piece)

print("\n" + "=" * 60)
print("Results:")
print("=" * 60)
print(f"Baseline params: beta={b_beta:.6f}, rho={b_rho:.6f}")
print(f"Baseline metrics: R²={r2_base:.4f}, MAE={mae_base:.4f}, RMSE={rmse_base:.4f}")
print(f"\nPiecewise params:")
print(f"  Pre-{int(BREAK_YEAR)}: beta={p_beta1:.6f}, rho={p_rho1:.6f}")
print(f"  Post-{int(BREAK_YEAR)}: beta={p_beta2:.6f}, rho={p_rho2:.6f}")
print(f"Piecewise metrics: R²={r2_piece:.4f}, MAE={mae_piece:.4f}, RMSE={rmse_piece:.4f}")
print("=" * 60)

# -----------------------------
# Save results
# -----------------------------
summary = {
    "data": {
        "csv": CSV_PATH,
        "start_year": int(t0),
        "end_year": int(t1),
        "n_points": len(years)
    },
    "break_year": int(BREAK_YEAR),
    "baseline": {
        "params": {
            "beta": float(b_beta),
            "rho": float(b_rho)
        },
        "metrics": {
            "R2": float(r2_base),
            "MAE": float(mae_base),
            "RMSE": float(rmse_base)
        }
    },
    "piecewise": {
        "params": {
            "beta_pre": float(p_beta1),
            "beta_post": float(p_beta2),
            "rho_pre": float(p_rho1),
            "rho_post": float(p_rho2)
        },
        "metrics": {
            "R2": float(r2_piece),
            "MAE": float(mae_piece),
            "RMSE": float(rmse_piece)
        }
    },
    "notes": {
        "model": "dy/dt = beta(t) * y*(1-y) - rho(t)*y",
        "piecewise": f"beta and rho change at {int(BREAK_YEAR)}"
    }
}

# Save JSON
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary: {OUT_JSON}")

# Save predictions CSV
pred_df = pd.DataFrame({
    "year": years,
    "obs_share": y_obs,
    "pred_baseline": y_base,
    "pred_piecewise": y_piece
})
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
pred_df.to_csv(OUT_CSV, index=False)
print(f"Saved predictions: {OUT_CSV}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 7))
plt.scatter(years, y_obs, s=25, label="Observed (membership share)", color="black", alpha=0.7)
plt.plot(t_base, y_base_dense, linewidth=2, label=f"Model (baseline, $R^2$={r2_base:.3f})")
plt.plot(t_piece, y_piece_dense, linewidth=2, label=f"Model (piecewise, $R^2$={r2_piece:.3f})")
plt.axvline(BREAK_YEAR, linestyle="--", linewidth=1, color="gray", alpha=0.7, label=f"Break = {int(BREAK_YEAR)}")
plt.xlabel("Year")
plt.ylabel("Share")
plt.ylim(0.0, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
print(f"Saved figures: {OUT_PNG}, {OUT_PDF}")

print("\nDone!")


