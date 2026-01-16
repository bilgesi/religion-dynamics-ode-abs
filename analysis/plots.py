"""
Plotting functions for scenario time series and bridge figures.

Generates publication-quality figures comparing ODE and ABS trajectories.
Analysis module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_bridge_scatter(
    metrics_rows: List[Dict],
    outpath: Path,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Scatter: x=MAE, y=R^2.
    No title (paper style). Optionally set axis limits for zoomed views.
    """
    maes = np.array([row["mean_mae"] for row in metrics_rows], dtype=float)
    r2s = np.array([row["mean_r2"] for row in metrics_rows], dtype=float)

    plt.figure()
    plt.scatter(maes, r2s)
    plt.xlabel("MAE (mean across religions)")
    plt.ylabel("R² (mean across religions)")

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=250)
    plt.close()


def plot_scenario_timeseries(ode_out: Dict, abs_out: Dict, religions: List[int], outpath: Path) -> None:
    """
    Simple time series plot (no title; captions belong in paper).
    Plots total shares (B+M+P)/N for each religion for ABS only.
    """
    # ABS totals
    t = np.array(abs_out["t"], dtype=float)
    S = np.array(abs_out["S"], dtype=float)
    N = S.copy()
    for r in religions:
        N += np.array(abs_out["B"][str(r)], dtype=float)
        N += np.array(abs_out["M"][str(r)], dtype=float)
        N += np.array(abs_out["P"][str(r)], dtype=float)

    plt.figure()
    for r in religions:
        Br = np.array(abs_out["B"][str(r)], dtype=float)
        Mr = np.array(abs_out["M"][str(r)], dtype=float)
        Pr = np.array(abs_out["P"][str(r)], dtype=float)
        y_abs = (Br + Mr + Pr) / np.maximum(N, 1e-12)
        plt.plot(t, y_abs, label=f"r={r}")

    plt.xlabel("Time")
    plt.ylabel("Total share (B+M+P)/N")
    plt.legend(fontsize=8)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    # Also save PDF version
    pdf_path = outpath.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def plot_scenario_A_timeseries(ode_out: Dict, abs_out: Dict, religions: List[int], outpath: Path) -> None:
    """
    Special formatting for Scenario A time series plot.
    Custom styling: Weeks on x-axis, specific ranges, grid, line styles, no legend.
    """
    # ABS totals
    t = np.array(abs_out["t"], dtype=float)
    S = np.array(abs_out["S"], dtype=float)
    N = S.copy()
    for r in religions:
        N += np.array(abs_out["B"][str(r)], dtype=float)
        N += np.array(abs_out["M"][str(r)], dtype=float)
        N += np.array(abs_out["P"][str(r)], dtype=float)

    fig, ax = plt.subplots()
    
    # Plot with specific styles for r=1 and r=2
    for r in sorted(religions):
        Br = np.array(abs_out["B"][str(r)], dtype=float)
        Mr = np.array(abs_out["M"][str(r)], dtype=float)
        Pr = np.array(abs_out["P"][str(r)], dtype=float)
        y_abs = (Br + Mr + Pr) / np.maximum(N, 1e-12)
        
        if r == 1:
            # r=1: solid black, linewidth=2
            ax.plot(t, y_abs, color='black', linestyle='-', linewidth=2)
        elif r == 2:
            # r=2: dashed black, linewidth=2
            ax.plot(t, y_abs, color='black', linestyle='--', linewidth=2)

    # Set labels with font size 16
    ax.set_xlabel("Weeks", fontsize=16)
    ax.set_ylabel("Portion of the population", fontsize=16)
    
    # Set ranges
    ax.set_xlim(0, 216)
    ax.set_ylim(0, 1)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid: both axes, alpha=0.25, black
    ax.grid(True, alpha=0.25, color='black')
    
    # Set ticks: x-axis every 27, y-axis every 0.1
    ax.set_xticks(np.arange(0, 217, 27))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # No legend
    # (already not added)
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    # Also save PDF version
    pdf_path = outpath.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def plot_scenario_B_timeseries(ode_out: Dict, abs_out: Dict, religions: List[int], outpath: Path) -> None:
    """
    Special formatting for Scenario B time series plot.
    Custom styling: Weeks on x-axis, specific ranges, grid, line styles, no legend.
    r=1: solid black, r=2,3,4,5,6: dashed with different colors.
    """
    # ABS totals
    t = np.array(abs_out["t"], dtype=float)
    S = np.array(abs_out["S"], dtype=float)
    N = S.copy()
    for r in religions:
        N += np.array(abs_out["B"][str(r)], dtype=float)
        N += np.array(abs_out["M"][str(r)], dtype=float)
        N += np.array(abs_out["P"][str(r)], dtype=float)

    fig, ax = plt.subplots()
    
    # Color palette for r=2,3,4,5,6 (dashed lines)
    colors = {
        2: '#1f77b4',  # blue
        3: '#ff7f0e',  # orange
        4: '#2ca02c',  # green
        5: '#d62728',  # red
        6: '#9467bd',  # purple
    }
    
    # Plot with specific styles
    for r in sorted(religions):
        Br = np.array(abs_out["B"][str(r)], dtype=float)
        Mr = np.array(abs_out["M"][str(r)], dtype=float)
        Pr = np.array(abs_out["P"][str(r)], dtype=float)
        y_abs = (Br + Mr + Pr) / np.maximum(N, 1e-12)
        
        if r == 1:
            # r=1: solid black, linewidth=2
            ax.plot(t, y_abs, color='black', linestyle='-', linewidth=2)
        elif r in [2, 3, 4, 5, 6]:
            # r=2,3,4,5,6: dashed with different colors, linewidth=2
            ax.plot(t, y_abs, color=colors[r], linestyle='--', linewidth=2)

    # Set labels with font size 16
    ax.set_xlabel("Weeks", fontsize=16)
    ax.set_ylabel("Portion of the population", fontsize=16)
    
    # Set ranges
    ax.set_xlim(0, 216)
    ax.set_ylim(0, 1)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid: both axes, alpha=0.25, black
    ax.grid(True, alpha=0.25, color='black')
    
    # Set ticks: x-axis every 27, y-axis every 0.1
    ax.set_xticks(np.arange(0, 217, 27))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # No legend
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    # Also save PDF version
    pdf_path = outpath.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def plot_scenario_C_timeseries(ode_out: Dict, abs_out: Dict, religions: List[int], outpath: Path) -> None:
    """
    Special formatting for Scenario C time series plot.
    Custom styling: Weeks on x-axis, specific ranges, grid, line styles, no legend.
    r=1: solid black, r=2: dashed black (like Scenario A).
    Note: Scenario C has longer time range (400 weeks) but we'll use 0-400 for x-axis.
    """
    # ABS totals
    t = np.array(abs_out["t"], dtype=float)
    S = np.array(abs_out["S"], dtype=float)
    N = S.copy()
    for r in religions:
        N += np.array(abs_out["B"][str(r)], dtype=float)
        N += np.array(abs_out["M"][str(r)], dtype=float)
        N += np.array(abs_out["P"][str(r)], dtype=float)

    fig, ax = plt.subplots()
    
    # Plot with specific styles for r=1 and r=2 (same as Scenario A)
    for r in sorted(religions):
        Br = np.array(abs_out["B"][str(r)], dtype=float)
        Mr = np.array(abs_out["M"][str(r)], dtype=float)
        Pr = np.array(abs_out["P"][str(r)], dtype=float)
        y_abs = (Br + Mr + Pr) / np.maximum(N, 1e-12)
        
        if r == 1:
            # r=1: solid black, linewidth=2
            ax.plot(t, y_abs, color='black', linestyle='-', linewidth=2)
        elif r == 2:
            # r=2: dashed black, linewidth=2
            ax.plot(t, y_abs, color='black', linestyle='--', linewidth=2)

    # Set labels with font size 16
    ax.set_xlabel("Weeks", fontsize=16)
    ax.set_ylabel("Portion of the population", fontsize=16)
    
    # Set ranges: x-axis 0-400 (Scenario C has longer time), y-axis 0-1
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid: both axes, alpha=0.25, black
    ax.grid(True, alpha=0.25, color='black')
    
    # Set ticks: x-axis every 50 (for 0-400 range), y-axis every 0.1
    ax.set_xticks(np.arange(0, 401, 50))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # No legend
    
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    # Also save PDF version
    pdf_path = outpath.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()
