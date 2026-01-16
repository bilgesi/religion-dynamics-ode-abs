"""
Run Scenario C (transient minority dynamics) with ODE and ABS.

Generates time series data and figures for stable coexistence regime.
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
from analysis.plots import plot_scenario_C_timeseries
from src.ode.solve_ode import run_ode
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
    print("Running Scenario C...", flush=True)
    print("=" * 60, flush=True)
    
    cfg = DEFAULT_CONFIG
    ensure_dirs(cfg.outputs_dir, cfg.runs_dir, cfg.figs_dir)

    key = "C"
    print(f"\nScenario {key} starting...", flush=True)
    
    # Fixed t_star (determined from crossing time y2>=0.2)
    t_star = 102.7  # weeks
    
    # Run with piecewise beta0
    print(f"  Running with piecewise beta0 (t_star={t_star:.1f}, beta_pre=0.75, beta_post=0.25)...", flush=True)
    params, init = SCENARIOS[key]()
    params.dt = cfg.dt
    params.t_max = 400.0
    
    # Set piecewise parameters for r=2
    params.beta0[2] = 0.75  # Base value (used as fallback)
    params.t_crash = {2: t_star}
    params.beta0_pre = {2: 0.75}
    params.beta0_post = {2: 0.25}
    
    S0 = float(init["S0"])
    B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
    M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
    P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}

    print(f"  Solving ODE...", flush=True)
    seeding_events = None  # Scenario C has no scheduled seeding
    ode_out = run_ode(params, S0, B0, M0, P0)
    
    print(f"  ODE solved. Running ABS simulations ({cfg.abs_replicates} replicates)...", flush=True)
    abs_mean = _mean_abs_over_reps(params, init, reps=cfg.abs_replicates, seed=cfg.random_seed + 100, seeding_events=seeding_events)

    # Compute metrics from ABS output
    t_abs = np.array(abs_mean["t"])
    S_abs = np.array(abs_mean["S"])
    N_abs = S_abs.copy()
    for r in params.religions:
        N_abs += np.array(abs_mean["B"][str(r)]) + np.array(abs_mean["M"][str(r)]) + np.array(abs_mean["P"][str(r)])
    
    y2_abs = (np.array(abs_mean["B"]["2"]) + np.array(abs_mean["M"]["2"]) + np.array(abs_mean["P"]["2"])) / np.maximum(N_abs, 1e-12)
    
    # Find peak
    peak_idx = np.argmax(y2_abs)
    y2_peak = float(y2_abs[peak_idx])
    t_peak = float(t_abs[peak_idx])
    
    # Find y2 at t_star (interpolate if needed)
    star_idx = np.searchsorted(t_abs, t_star)
    if star_idx >= len(t_abs):
        star_idx = len(t_abs) - 1
    y2_at_star = float(y2_abs[star_idx])
    
    # Final y2 at T=400
    y2_final = float(y2_abs[-1])

    print(f"  Saving results...", flush=True)
    save_json({"ode": ode_out, "abs_mean": abs_mean}, cfg.runs_dir / f"scenario_{key}.json")
    
    plot_scenario_C_timeseries(ode_out, abs_mean, params.religions, cfg.figs_dir / f"fig_scenario_{key}.png")
    
    print(f"Scenario {key} completed!", flush=True)
    print(f"  y2_peak: {y2_peak:.4f} at t_peak = {t_peak:.1f} weeks", flush=True)
    print(f"  y2 at t_star ({t_star:.1f} weeks): {y2_at_star:.4f}", flush=True)
    print(f"  y2_final (T=400): {y2_final:.4f}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("Scenario C completed!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
