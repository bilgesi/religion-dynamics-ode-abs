"""
Run Scenario B (sequential strain invasions) with ODE and ABS.

Generates time series data and figures for multi-strain transient dynamics.
Active experiment script.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict
import numpy as np
from src.abs.init_population import init_population_from_counts
from src.abs.sim_abs import run_abs
from analysis.plots import plot_scenario_B_timeseries
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


def main():
    print("=" * 60, flush=True)
    print("Running Scenario B...", flush=True)
    print("=" * 60, flush=True)
    
    cfg = DEFAULT_CONFIG
    ensure_dirs(cfg.outputs_dir, cfg.runs_dir, cfg.figs_dir)

    key = "B"
    print(f"\nScenario {key} starting...", flush=True)
    params, init = SCENARIOS[key]()
    params.dt = cfg.dt
    params.t_max = 216.0

    S0 = float(init["S0"])
    B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
    M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
    P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}

    print(f"  Solving ODE...", flush=True)
    seeding_events = [
        (0.0, 2, {"B": 50, "M": 100, "P": 0}),
        (16.0, 3, {"B": 50, "M": 100, "P": 0}),
        (48.0, 4, {"B": 50, "M": 100, "P": 0}),
        (112.0, 5, {"B": 50, "M": 100, "P": 0}),
        (167.0, 6, {"B": 50, "M": 100, "P": 0}),
    ]
    ode_out = run_ode_with_seeding(params, S0, B0, M0, P0, seeding_events)
    
    print(f"  ODE solved. Running ABS simulations ({cfg.abs_replicates} replicates)...", flush=True)
    abs_mean = _mean_abs_over_reps(params, init, reps=cfg.abs_replicates, seed=cfg.random_seed + 100, seeding_events=seeding_events)

    print(f"  Saving results...", flush=True)
    save_json({"ode": ode_out, "abs_mean": abs_mean}, cfg.runs_dir / f"scenario_{key}.json")
    
    plot_scenario_B_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
    
    # Calculate and print peak/final metrics for cults (r=2..6)
    print("\n" + "=" * 60, flush=True)
    print("Cult Metrics (r=2..6):", flush=True)
    print("=" * 60, flush=True)
    t_abs = np.array(abs_mean["t"])
    S = np.array(abs_mean["S"])
    N_total = S + sum(np.array(abs_mean["B"][str(rel)]) + np.array(abs_mean["M"][str(rel)]) + np.array(abs_mean["P"][str(rel)]) for rel in params.religions)
    
    for r in [2, 3, 4, 5, 6]:
        B_r = np.array(abs_mean["B"][str(r)])
        M_r = np.array(abs_mean["M"][str(r)])
        P_r = np.array(abs_mean["P"][str(r)])
        total_r = B_r + M_r + P_r
        share_r = total_r / (N_total + 1e-9)  # Add small epsilon to avoid division by zero
        
        peak_idx = np.argmax(share_r)
        peak_share = float(share_r[peak_idx])
        peak_time = float(t_abs[peak_idx])
        final_share = float(share_r[-1])
        
        # Calculate max(M_r)/N
        M_ratio = M_r / (N_total + 1e-9)
        max_M_ratio = float(np.max(M_ratio))
        
        # Calculate max(lambda_r)
        from model.context import beta_r
        lambdas_r = np.array([beta_r(float(t), r, params) * (float(M_r[i]) / (N_total[i] + 1e-9)) for i, t in enumerate(t_abs)])
        max_lambda = float(np.max(lambdas_r))
        
        print(f"  r={r}: peak_share={peak_share:.4f}, peak_time={peak_time:.1f}, final_share={final_share:.4f}", flush=True)
        print(f"         max(M_r)/N={max_M_ratio:.4f}, max(lambda_r)={max_lambda:.4f}", flush=True)
    
    print(f"Scenario {key} completed!", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("Scenario B completed!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
