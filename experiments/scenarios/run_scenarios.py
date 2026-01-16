"""
Run all stylized scenarios (A, B, C) with ODE and ABS.

Generates time series data and publication figures for scenario comparison.
Active experiment script.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.abs.init_population import init_population_from_counts
from src.abs.sim_abs import run_abs
from analysis.plots import plot_scenario_timeseries, plot_scenario_A_timeseries, plot_scenario_B_timeseries, plot_scenario_C_timeseries
from src.ode.solve_ode import run_ode
from src.ode.solve_ode_with_seeding import run_ode_with_seeding
from src.scenarios import SCENARIOS
from src.utils import ensure_dirs, save_json
from src.config import DEFAULT_CONFIG


def _mean_abs_over_reps(params, init, reps: int, seed: int, seeding_events=None) -> Dict:
    rng = np.random.default_rng(seed)
    S0 = int(init["S0"])
    B0 = {int(k): int(v) for k, v in init["B0"].items()}
    M0 = {int(k): int(v) for k, v in init["M0"].items()}
    P0 = {int(k): int(v) for k, v in init["P0"].items()}

    acc = None
    for rep in range(reps):
        if rep % 5 == 0:
            print(f"  ABS replicate {rep+1}/{reps}...", flush=True)
        # Use dict copies to prevent mutation issues (defensive programming)
        roles, rel_ids = init_population_from_counts(S0, dict(B0), dict(M0), dict(P0), params.religions, rng)
        out = run_abs(params, roles, rel_ids, seed=int(rng.integers(1, 10**9)), seeding_events=seeding_events)
        if acc is None:
            acc = out
        else:
            acc["S"] = (np.array(acc["S"]) + np.array(out["S"])).tolist()
            for r in params.religions:
                for key in ["B", "M", "P"]:
                    acc[key][str(r)] = (np.array(acc[key][str(r)]) + np.array(out[key][str(r)])).tolist()

    assert acc is not None
    mean = {"t": acc["t"], "S": (np.array(acc["S"]) / reps).tolist(), "B": {}, "M": {}, "P": {}}
    for r in params.religions:
        mean["B"][str(r)] = (np.array(acc["B"][str(r)]) / reps).tolist()
        mean["M"][str(r)] = (np.array(acc["M"][str(r)]) / reps).tolist()
        mean["P"][str(r)] = (np.array(acc["P"][str(r)]) / reps).tolist()
    return mean


def run_all() -> None:
    print("=" * 60, flush=True)
    print("Starting scenario runs...", flush=True)
    print("=" * 60, flush=True)
    
    cfg = DEFAULT_CONFIG
    ensure_dirs(cfg.outputs_dir, cfg.runs_dir, cfg.figs_dir)

    for key in ["A", "B", "C"]:
        print(f"\nScenario {key} starting...", flush=True)
        params, init = SCENARIOS[key]()
        params.dt = cfg.dt
        if key == "A":
            params.t_max = 216.0  # 216 weeks for Scenario A
        elif key == "B":
            params.t_max = 216.0  # 216 weeks for Scenario B (same as A)
        elif key == "C":
            params.t_max = 400.0  # allow longer for steady scenario
        else:
            params.t_max = cfg.t_max

        S0 = float(init["S0"])
        B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
        M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
        P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}

        print(f"  Solving ODE...", flush=True)
        seeding_events = None
        if key == "B":
            # Scenario B: Use scheduled seeding for strains r=2..6
            # Seeding times: r2=0, r3=16, r4=48, r5=112, r6=167 (weeks)
            # Seeding with M=100 to reach ~0.2 peak (B=50, M=100)
            seeding_events = [
                (0.0, 2, {"B": 50, "M": 100, "P": 0}),
                (16.0, 3, {"B": 50, "M": 100, "P": 0}),
                (48.0, 4, {"B": 50, "M": 100, "P": 0}),
                (112.0, 5, {"B": 50, "M": 100, "P": 0}),
                (167.0, 6, {"B": 50, "M": 100, "P": 0}),
            ]
            ode_out = run_ode_with_seeding(params, S0, B0, M0, P0, seeding_events)
        else:
            ode_out = run_ode(params, S0, B0, M0, P0)
        print(f"  ODE solved. Running ABS simulations ({cfg.abs_replicates} replicates)...", flush=True)
        abs_mean = _mean_abs_over_reps(params, init, reps=cfg.abs_replicates, seed=cfg.random_seed + 100, seeding_events=seeding_events)

        print(f"  Saving results...", flush=True)
        save_json({"ode": ode_out, "abs_mean": abs_mean}, cfg.runs_dir / f"scenario_{key}.json")
        # Use special formatting for Scenario A, B, and C
        if key == "A":
            plot_scenario_A_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        elif key == "B":
            plot_scenario_B_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        elif key == "C":
            plot_scenario_C_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        else:
            plot_scenario_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        print(f"Scenario {key} completed!", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("All scenarios completed!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    print("Script started!", flush=True)
    run_all()
