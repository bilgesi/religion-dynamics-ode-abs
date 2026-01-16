#!/usr/bin/env python3
"""
Plot New Zealand "none" group fit results.
Generates fig_hist_nz.pdf with baseline and piecewise fits.
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = Path("_legacy/out/nz_none_predictions.csv")
SUMMARY_JSON = Path("_legacy/out/nz_none_fit_summary.json")
OUT_PDF = Path("outputs/figures/fig_hist_nz.pdf")

def main():
    # Load data
    if not PRED_CSV.exists():
        print(f"ERROR: Predictions file not found: {PRED_CSV}", file=__import__("sys").stderr)
        __import__("sys").exit(1)
    
    if not SUMMARY_JSON.exists():
        print(f"ERROR: Summary file not found: {SUMMARY_JSON}", file=__import__("sys").stderr)
        __import__("sys").exit(1)
    
    pred = pd.read_csv(PRED_CSV)
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summ = json.load(f)
    
    # Extract data
    x = pred["year"]
    y_obs = pred["obs_share"]
    y_base = pred["pred_baseline"]
    y_piece = pred["pred_piecewise"]
    
    # Extract metrics
    r2_base = summ["baseline"].get("R2", 0.0)
    r2_piece = summ["piecewise"].get("R2", 0.0)
    break_year = summ["piecewise"].get("break_year", None)
    
    # Create figure (no title - LaTeX will handle panel labels)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    
    # Plot observed points
    ax.scatter(x, y_obs, s=50, label="Observed", color="black", alpha=0.7, zorder=3)
    
    # Plot baseline curve
    ax.plot(x, y_base, linewidth=2, label=f"Baseline ($R^2$={r2_base:.3f})", zorder=2)
    
    # Plot piecewise curve
    if break_year is not None:
        ax.plot(x, y_piece, linewidth=2, linestyle="--", 
                label=f"Piecewise ($R^2$={r2_piece:.3f}, break={int(break_year)})", zorder=2)
        # Vertical line at break year
        ax.axvline(break_year, linestyle=":", linewidth=1, color="gray", alpha=0.5, zorder=1)
    else:
        ax.plot(x, y_piece, linewidth=2, linestyle="--", 
                label=f"Piecewise ($R^2$={r2_piece:.3f})", zorder=2)
    
    # Labels (no title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    
    # Styling
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    
    # Save
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {OUT_PDF}")

if __name__ == "__main__":
    main()
