"""
Run Scenarios B and C with ODE only (no ABS) for quick figure generation.

Skips computationally expensive ABS replicates. Active experiment script.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from analysis.plots import plot_scenario_B_timeseries, plot_scenario_C_timeseries
from src.ode.solve_ode import run_ode
from src.ode.solve_ode_with_seeding import run_ode_with_seeding
from src.scenarios import SCENARIOS
from src.utils import ensure_dirs
from src.config import DEFAULT_CONFIG


def main():
    print("=" * 60, flush=True)
    print("Running scenarios B and C (ODE only, no ABS)...", flush=True)
    print("=" * 60, flush=True)
    
    cfg = DEFAULT_CONFIG
    ensure_dirs(cfg.outputs_dir, cfg.runs_dir, cfg.figs_dir)

    for key in ["B", "C"]:
        print(f"\nScenario {key} starting...", flush=True)
        params, init = SCENARIOS[key]()
        params.dt = cfg.dt
        if key == "B":
            params.t_max = 216.0
        elif key == "C":
            params.t_max = 400.0

        S0 = float(init["S0"])
        B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
        M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
        P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}

        print(f"  Solving ODE...", flush=True)
        if key == "B":
            seeding_events = [
                (0.0, 2, {"B": 180, "M": 25, "P": 0}),
                (16.0, 3, {"B": 180, "M": 25, "P": 0}),
                (48.0, 4, {"B": 180, "M": 25, "P": 0}),
                (112.0, 5, {"B": 180, "M": 25, "P": 0}),
                (167.0, 6, {"B": 180, "M": 25, "P": 0}),
            ]
            ode_out = run_ode_with_seeding(params, S0, B0, M0, P0, seeding_events)
        else:
            ode_out = run_ode(params, S0, B0, M0, P0)
        
        print(f"  ODE solved. Creating figure (ODE only, no ABS)...", flush=True)
        
        # Create dummy ABS output with ODE data (for plotting compatibility)
        abs_mean = {
            "t": ode_out["t"],
            "S": ode_out["S"],
            "B": ode_out["B"],
            "M": ode_out["M"],
            "P": ode_out["P"],
        }
        
        if key == "B":
            plot_scenario_B_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        elif key == "C":
            plot_scenario_C_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
        
        print(f"Scenario {key} completed!", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("Scenarios B and C completed (ODE only)!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
