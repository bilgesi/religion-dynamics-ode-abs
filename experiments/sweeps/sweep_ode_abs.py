"""
Parameter sweep comparing ODE and ABS trajectories.

Generates bridge figure data by randomizing parameters and computing
MAE/R-squared between ODE and ABS outputs. Active experiment script.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.abs.init_population import init_population_from_counts
from src.abs.sim_abs import run_abs
from analysis.metrics import compare_ode_abs_totals
from analysis.plots import plot_bridge_scatter
from src.model.params import ModelParams, copy_with_dt
from src.ode.solve_ode import run_ode
from src.scenarios import scenario_A
from src.utils import ensure_dirs, save_json
from src.config import DEFAULT_CONFIG


def _randomize_params(base: ModelParams, rng: np.random.Generator) -> ModelParams:
    """
    Create a random parameter set around base, for the bridge figure sweep.
    Keeps the same rule set; changes values only.
    """
    p = copy_with_dt(base, dt=base.dt, t_max=base.t_max)

    # Random multiplicative jitter (bounded)
    for r in p.religions:
        p.beta0[r] *= float(rng.uniform(0.7, 1.3))
        p.q[r] = float(np.clip(p.q[r] * rng.uniform(0.7, 1.3), 0.0, 1.0))
        p.sigma[r] *= float(rng.uniform(0.7, 1.3))
        p.kappa[r] *= float(rng.uniform(0.7, 1.3))
        p.tauB[r] *= float(rng.uniform(0.7, 1.3))
        p.tauM[r] *= float(rng.uniform(0.7, 1.3))
        p.rhoB[r] *= float(rng.uniform(0.7, 1.3))
        p.rhoM[r] *= float(rng.uniform(0.7, 1.3))
        p.rhoP[r] *= float(rng.uniform(0.7, 1.3))

    # Mutation jitter (if exists)
    if p.nu is not None:
        for rr in list(p.nu.keys()):
            for ll in list(p.nu[rr].keys()):
                p.nu[rr][ll] *= float(rng.uniform(0.7, 1.3))

    return p


def run_sweep(n_cases: int = 30, replicates: int = 20, out_csv: bool = False) -> None:
    print("=" * 60, flush=True)
    print(f"Starting ODE-ABS sweep: {n_cases} cases, {replicates} replicates each", flush=True)
    print(f"Total ABS simulations: {n_cases * replicates}", flush=True)
    print("=" * 60, flush=True)
    
    cfg = DEFAULT_CONFIG
    ensure_dirs(cfg.outputs_dir, cfg.runs_dir, cfg.figs_dir)

    base_params, base_init = scenario_A()  # base template for sweep
    base_params.dt = cfg.dt
    base_params.t_max = cfg.t_max

    rng = np.random.default_rng(cfg.random_seed)

    metrics_rows: List[Dict] = []
    for i in range(n_cases):
        print(f"\nCase {i+1}/{n_cases} starting...", flush=True)
        params_i = _randomize_params(base_params, rng)

        # initial conditions (fixed here; could also jitter if desired)
        S0 = int(base_init["S0"])
        B0 = {int(k): int(v) for k, v in base_init["B0"].items()}
        M0 = {int(k): int(v) for k, v in base_init["M0"].items()}
        P0 = {int(k): int(v) for k, v in base_init["P0"].items()}

        print(f"  Solving ODE...", flush=True)
        ode_out = run_ode(params_i, float(S0), {r: float(B0.get(r, 0)) for r in params_i.religions},
                          {r: float(M0.get(r, 0)) for r in params_i.religions},
                          {r: float(P0.get(r, 0)) for r in params_i.religions})
        print(f"  ODE solved. Running ABS simulations ({replicates} replicates)...", flush=True)

        # ABS mean over replicates
        abs_acc = None
        for rep in range(replicates):
            if rep % 5 == 0 or rep == 0:
                print(f"    ABS replicate {rep+1}/{replicates}...", flush=True)
            roles, rel_ids = init_population_from_counts(S0, B0, M0, P0, params_i.religions, rng)
            abs_out = run_abs(params_i, roles, rel_ids, seed=int(rng.integers(1, 10**9)))
            if abs_acc is None:
                abs_acc = abs_out
            else:
                # accumulate
                abs_acc["S"] = (np.array(abs_acc["S"]) + np.array(abs_out["S"])).tolist()
                for r in params_i.religions:
                    for key in ["B", "M", "P"]:
                        abs_acc[key][str(r)] = (np.array(abs_acc[key][str(r)]) + np.array(abs_out[key][str(r)])).tolist()

        # average
        assert abs_acc is not None
        abs_mean = {"t": abs_acc["t"], "S": (np.array(abs_acc["S"]) / replicates).tolist(), "B": {}, "M": {}, "P": {}}
        for r in params_i.religions:
            abs_mean["B"][str(r)] = (np.array(abs_acc["B"][str(r)]) / replicates).tolist()
            abs_mean["M"][str(r)] = (np.array(abs_acc["M"][str(r)]) / replicates).tolist()
            abs_mean["P"][str(r)] = (np.array(abs_acc["P"][str(r)]) / replicates).tolist()

        print(f"  Computing metrics...", flush=True)
        metrics = compare_ode_abs_totals(ode_out, abs_mean, params_i.religions)
        row = {
            "case": i,
            "mean_mae": metrics["mean_mae"],
            "mean_r2": metrics["mean_r2"],
            "params": {
                "b": params_i.b,
                "mu": params_i.mu,
                "beta0": params_i.beta0,
                "q": params_i.q,
                "sigma": params_i.sigma,
                "kappa": params_i.kappa,
                "tauB": params_i.tauB,
                "tauM": params_i.tauM,
                "rhoB": params_i.rhoB,
                "rhoM": params_i.rhoM,
                "rhoP": params_i.rhoP,
                "nu": params_i.nu,
                "dt": params_i.dt,
                "t_max": params_i.t_max,
            }
        }
        metrics_rows.append(row)
        print(f"  Case {i+1} completed! MAE={metrics['mean_mae']:.6f}, R²={metrics['mean_r2']:.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Sweep completed! Saving results...", flush=True)
    # Save metrics
    save_json({"metrics": metrics_rows}, cfg.runs_dir / "bridge_metrics.json")
    print("  Metrics saved to bridge_metrics.json", flush=True)

    # Plot bridge figures (full and zoomed views)
    print("  Generating bridge scatter plots...", flush=True)
    # Full view (as Teddy requested)
    plot_bridge_scatter(
        metrics_rows,
        cfg.figs_dir / "fig_bridge_full.png",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    print("  Full view saved to fig_bridge_full.png", flush=True)
    
    # Zoomed-in view (shows the "amazing cluster" clearly)
    plot_bridge_scatter(
        metrics_rows,
        cfg.figs_dir / "fig_bridge_zoom.png",
        xlim=(0.0, 0.02),
        ylim=(0.9, 1.0),
    )
    print("  Zoomed view saved to fig_bridge_zoom.png", flush=True)
    print("=" * 60, flush=True)
    print("All done!", flush=True)


if __name__ == "__main__":
    print("Script started!", flush=True)
    run_sweep(n_cases=30, replicates=20)
