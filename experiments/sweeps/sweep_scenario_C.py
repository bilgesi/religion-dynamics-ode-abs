# experiments/sweep_scenario_C.py
"""
Parameter sweep for Scenario C to find parameters where r=2 is transient:
- Rises to at least 0.02
- Dies out to <= 0.002 at the end
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.model.params import ModelParams
from src.ode.solve_ode import run_ode
from src.scenarios import scenario_C


def compute_metrics(t: np.ndarray, y2: np.ndarray) -> Dict[str, float]:
    """Compute peak and final mean for religion 2."""
    peak = float(np.max(y2))
    
    # Final mean over last 20% of time
    last_20_percent_idx = int(len(t) * 0.8)
    final_mean = float(np.mean(y2[last_20_percent_idx:]))
    
    return {"peak": peak, "final_mean": final_mean}


def sweep_scenario_C() -> Tuple[Dict, List[Dict]]:
    """
    Sweep parameters for Scenario C.
    Returns: (best_params_dict, all_results_list)
    """
    # Get base scenario
    params_base, init_base = scenario_C()
    
    # Base values for r=2
    beta0_base = params_base.beta0[2]
    rhoM_base = params_base.rhoM[2]
    rhoB_base = params_base.rhoB[2]
    
    # Search ranges (multiplicative factors)
    # Expand beta0 range significantly to allow growth to 0.02
    beta0_factors = [1.5, 2.0, 2.5, 3.0]
    # Reduce rhoM/rhoB to allow growth before dying out
    rhoM_factors = [0.5, 1.0, 1.5, 2.0]
    rhoB_factors = [0.5, 1.0, 1.5]
    
    # Use very small mutation to allow r=2 to grow from seed, or seed with more agents
    # Option 1: Keep tiny mutation rate
    # params_base.nu = {1: {2: 0.0001}, 2: {1: 0.0}}  # Very small mutation
    
    # Option 2: Disable mutation but seed with more agents
    params_base.nu = {1: {2: 0.0}, 2: {1: 0.0}}
    # Seed r=2 with small initial count (B2=100 to allow growth to 0.02)
    init_base["B0"][2] = 100
    init_base["M0"][2] = 0
    init_base["P0"][2] = 0
    
    S0 = float(init_base["S0"])
    B0 = {r: float(init_base["B0"].get(r, 0)) for r in params_base.religions}
    M0 = {r: float(init_base["M0"].get(r, 0)) for r in params_base.religions}
    P0 = {r: float(init_base["P0"].get(r, 0)) for r in params_base.religions}
    
    all_results = []
    best_candidate = None
    best_peak = -np.inf
    
    print("=" * 80)
    print("Scenario C Parameter Sweep")
    print("=" * 80)
    print(f"Base values: beta0[2]={beta0_base:.4f}, rhoM[2]={rhoM_base:.4f}, rhoB[2]={rhoB_base:.4f}")
    print(f"Search ranges:")
    print(f"  beta0[2] * {beta0_factors}")
    print(f"  rhoM[2]  * {rhoM_factors}")
    print(f"  rhoB[2]  * {rhoB_factors}")
    print(f"Acceptance criteria: peak >= 0.02, final <= 0.002")
    print("=" * 80)
    
    total_combinations = len(beta0_factors) * len(rhoM_factors) * len(rhoB_factors)
    current = 0
    
    for beta0_factor in beta0_factors:
        for rhoM_factor in rhoM_factors:
            for rhoB_factor in rhoB_factors:
                current += 1
                
                # Create modified params (create new dicts to avoid mutating base)
                beta0_new = dict(params_base.beta0)
                rhoM_new = dict(params_base.rhoM)
                rhoB_new = dict(params_base.rhoB)
                beta0_new[2] = beta0_base * beta0_factor
                rhoM_new[2] = rhoM_base * rhoM_factor
                rhoB_new[2] = rhoB_base * rhoB_factor
                
                params = ModelParams(
                    religions=list(params_base.religions),
                    b=params_base.b,
                    mu=params_base.mu,
                    beta0=beta0_new,
                    q=dict(params_base.q),
                    sigma=dict(params_base.sigma),
                    kappa=dict(params_base.kappa),
                    tauB=dict(params_base.tauB),
                    tauM=dict(params_base.tauM),
                    rhoB=rhoB_new,
                    rhoM=rhoM_new,
                    rhoP=dict(params_base.rhoP),
                    nu=params_base.nu,  # Already set to 0 above
                    context_enabled=params_base.context_enabled,
                    dt=params_base.dt,
                    t_max=params_base.t_max,
                )
                
                # Run ODE
                try:
                    ode_out = run_ode(params, S0, B0, M0, P0)
                    
                    # Extract r=2 trajectory
                    t = np.array(ode_out["t"], dtype=float)
                    B2 = np.array(ode_out["B"]["2"], dtype=float)
                    M2 = np.array(ode_out["M"]["2"], dtype=float)
                    P2 = np.array(ode_out["P"]["2"], dtype=float)
                    S = np.array(ode_out["S"], dtype=float)
                    
                    # Compute total population
                    N = S + B2 + M2 + P2
                    for r in params.religions:
                        if r != 2:
                            N += np.array(ode_out["B"][str(r)], dtype=float)
                            N += np.array(ode_out["M"][str(r)], dtype=float)
                            N += np.array(ode_out["P"][str(r)], dtype=float)
                    
                    # Compute share of r=2
                    y2 = (B2 + M2 + P2) / np.maximum(N, 1e-12)
                    
                    # Compute metrics
                    metrics = compute_metrics(t, y2)
                    peak = metrics["peak"]
                    final = metrics["final_mean"]
                    
                    # Check acceptance
                    accepted = (peak >= 0.02) and (final <= 0.002)
                    
                    result = {
                        "beta0_factor": beta0_factor,
                        "rhoM_factor": rhoM_factor,
                        "rhoB_factor": rhoB_factor,
                        "beta0": params.beta0[2],
                        "rhoM": params.rhoM[2],
                        "rhoB": params.rhoB[2],
                        "peak": peak,
                        "final_mean": final,
                        "accepted": accepted,
                    }
                    all_results.append(result)
                    
                    status = "ACCEPTED" if accepted else "rejected"
                    print(f"[{current}/{total_combinations}] {status} | "
                          f"beta0*{beta0_factor:.1f}, rhoM*{rhoM_factor:.1f}, rhoB*{rhoB_factor:.1f} | "
                          f"peak={peak:.4f}, final={final:.4f}")
                    
                    # Update best candidate
                    if accepted and peak > best_peak:
                        best_peak = peak
                        best_candidate = {
                            "beta0": params.beta0[2],
                            "rhoM": params.rhoM[2],
                            "rhoB": params.rhoB[2],
                            "beta0_factor": beta0_factor,
                            "rhoM_factor": rhoM_factor,
                            "rhoB_factor": rhoB_factor,
                            "peak": peak,
                            "final_mean": final,
                        }
                
                except Exception as e:
                    print(f"[{current}/{total_combinations}] ERROR: {e}")
                    all_results.append({
                        "beta0_factor": beta0_factor,
                        "rhoM_factor": rhoM_factor,
                        "rhoB_factor": rhoB_factor,
                        "error": str(e),
                    })
    
    print("=" * 80)
    if best_candidate is not None:
        print("BEST CANDIDATE:")
        print(f"  beta0[2] = {best_candidate['beta0']:.6f} (factor: {best_candidate['beta0_factor']:.1f})")
        print(f"  rhoM[2]  = {best_candidate['rhoM']:.6f} (factor: {best_candidate['rhoM_factor']:.1f})")
        print(f"  rhoB[2]  = {best_candidate['rhoB']:.6f} (factor: {best_candidate['rhoB_factor']:.1f})")
        print(f"  peak     = {best_candidate['peak']:.4f}")
        print(f"  final    = {best_candidate['final_mean']:.4f}")
    else:
        print("NO ACCEPTED CANDIDATES FOUND")
    print("=" * 80)
    
    return best_candidate, all_results


if __name__ == "__main__":
    best, all_results = sweep_scenario_C()
    
    # Save results
    import json
    from config import DEFAULT_CONFIG
    from utils import ensure_dirs
    
    ensure_dirs(DEFAULT_CONFIG.runs_dir)
    output_path = DEFAULT_CONFIG.runs_dir / "scenario_C_sweep.json"
    
    with open(output_path, "w") as f:
        json.dump({"best": best, "all_results": all_results}, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
