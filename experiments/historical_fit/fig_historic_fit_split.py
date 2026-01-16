#!/usr/bin/env python3
"""
Generate separate PDF figures for Sweden, Turkey (JW), and Finland.
Each figure is a single panel, suitable for LaTeX subfigure inclusion.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG: match with your file names ----------
CONFIG = [
    {
        "name": "sweden",
        "pred_csv": "outputs/runs/sweden_poc_predictions.csv",         # year, obs_share, pred_baseline, pred_piecewise
        "summary_json": "outputs/runs/sweden_poc_fit_summary.json",    # baseline/piecewise metrics + break year
        "out_pdf": "outputs/figures/fig_hist_sweden.pdf",
        "out_png": "outputs/figures/fig_hist_sweden.png",
    },
    {
        "name": "turkey",
        "pred_csv": "outputs/runs/jw_turkey_predictions.csv",          # Will be created if missing
        "summary_json": "outputs/runs/jw_turkey_fit_summary.json",
        "out_pdf": "outputs/figures/fig_hist_turkey.pdf",
        "out_png": "outputs/figures/fig_hist_turkey.png",
    },
    {
        "name": "newzealand",
        "pred_csv": "out/nz_none_predictions.csv",
        "summary_json": "out/nz_none_fit_summary.json",
        "out_pdf": "outputs/figures/fig_hist_newzealand.pdf",
        "out_png": "outputs/figures/fig_hist_newzealand.png",
    },
]
# ----------------------------------------------------------


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_one(pred_path, sum_path, out_pdf, out_png, name=None):
    pred = pd.read_csv(pred_path)
    summ = load_summary(sum_path)

    # Expected column names:
    # year, obs_share, pred_baseline, pred_piecewise
    # (change here if your column names differ)
    x = pred["year"]
    y_obs = pred["obs_share"]
    y_base = pred["pred_baseline"]
    y_piece = pred["pred_piecewise"]

    # Handle different JSON structures
    if "baseline" in summ:
        # Sweden format: baseline.metrics.R2
        if "metrics" in summ["baseline"]:
            r2_base = summ["baseline"]["metrics"].get("R2", 0.0)
            r2_piece = summ["piecewise"]["metrics"].get("R2", 0.0)
        else:
            # Finland format: baseline.R2
            r2_base = summ["baseline"].get("R2", 0.0)
            r2_piece = summ["piecewise"].get("R2", 0.0)
        break_year = summ.get("break_year") or summ["piecewise"].get("break_year", None)
    else:
        # Fallback for different structure
        r2_base = summ.get("R2", 0.0)
        r2_piece = summ.get("R2", 0.0)
        break_year = None

    # Same styling for all countries
    # Single panel size: ideal for LaTeX subfigure
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    # Historical data (black dots)
    ax.scatter(x, y_obs, s=28, label="Historical data", color="black", alpha=0.7, zorder=3)
    
    # Baseline (black solid line) - each country's own R² value
    ax.plot(x, y_base, linewidth=2, label=f"Baseline ($R^2$ = {r2_base:.3f})", color="black", linestyle="-", zorder=2)
    
    # Piecewise (black dashed line) - each country's own R² value
    ax.plot(x, y_piece, linewidth=2, label=f"Piecewise ($R^2$ = {r2_piece:.3f})", color="black", linestyle="--", zorder=2)

    # Break year vertical line (no legend)
    if break_year is not None:
        ax.axvline(break_year, linestyle=":", linewidth=1, color="gray", alpha=0.5, zorder=1)

    # X-axis: first to last value, 11 ticks (integer years only)
    x_min = float(x.min())
    x_max = float(x.max())
    ax.set_xlim(x_min, x_max)
    # Create 11 ticks but round to nearest integers
    xticks = np.linspace(x_min, x_max, 11)
    ax.set_xticks(xticks)
    # Format as integers
    ax.set_xticklabels([f'{int(round(t))}' for t in xticks])
    
    # Rotate x-axis labels to avoid overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Y-axis: 11 ticks
    y_min = min(float(y_obs.min()), float(y_base.min()), float(y_piece.min()))
    y_max = max(float(y_obs.max()), float(y_base.max()), float(y_piece.max()))
    ax.set_ylim(y_min * 0.98, y_max * 1.02)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 11))

    # Labels with fontsize 16
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Share", fontsize=16)
    
    # Remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    for cfg in CONFIG:
        pred_path = Path(cfg["pred_csv"])
        sum_path = Path(cfg["summary_json"])
        out_pdf = Path(cfg["out_pdf"])
        out_png = Path(cfg["out_png"])

        if not pred_path.exists():
            print(f"WARNING: Missing predictions: {pred_path}")
            print(f"  Skipping {cfg['name']}...")
            continue
        if not sum_path.exists():
            print(f"WARNING: Missing summary: {sum_path}")
            print(f"  Skipping {cfg['name']}...")
            continue

        plot_one(pred_path, sum_path, out_pdf, out_png, name=cfg["name"])
        print(f"Wrote: {out_pdf}")
        print(f"Wrote: {out_png}")

if __name__ == "__main__":
    main()
