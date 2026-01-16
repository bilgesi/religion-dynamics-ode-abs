"""
Generate ODE vs ABS case summary across multiple scenarios.

Computes comparison metrics for different parameter configurations and exports
to CSV. Active experiment script for paper artifact generation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.abs.init_population import init_population_from_counts
from src.abs.sim_abs import run_abs
from analysis.metrics import compare_ode_abs_totals
from src.model.params import ModelParams, copy_with_dt
from src.ode.solve_ode import run_ode
from src.ode.solve_ode_with_seeding import run_ode_with_seeding
from src.scenarios import SCENARIOS
from src.utils import ensure_dirs


def sample_params(params0: ModelParams, rng: np.random.Generator, case_type: str) -> ModelParams:
    """
    Create a perturbed copy of params0 for case variation.
    Perturbs key effective parameters (beta0, rhoB, rhoM, q) within reasonable ranges.
    """
    params = copy_with_dt(params0, dt=params0.dt, t_max=params0.t_max)
    
    # Random multiplicative jitter (bounded, similar to sweep_ode_abs.py)
    for r in params.religions:
        # Perturb beta0 (0.7x to 1.3x)
        params.beta0[r] *= float(rng.uniform(0.7, 1.3))
        # Perturb q (0.7x to 1.3x, clipped to [0, 1])
        params.q[r] = float(np.clip(params.q[r] * rng.uniform(0.7, 1.3), 0.0, 1.0))
        # Perturb rhoB (0.7x to 1.3x)
        params.rhoB[r] *= float(rng.uniform(0.7, 1.3))
        # Perturb rhoM (0.7x to 1.3x)
        params.rhoM[r] *= float(rng.uniform(0.7, 1.3))
    
    # Mutation jitter (if exists)
    if params.nu is not None:
        for rr in list(params.nu.keys()):
            for ll in list(params.nu[rr].keys()):
                params.nu[rr][ll] *= float(rng.uniform(0.7, 1.3))
    
    return params


def run_ode_for_case(
    params: ModelParams, 
    init: Dict, 
    seeding_events: Optional[List[Tuple[float, int, Dict[str, int]]]] = None
) -> Dict:
    """Run ODE simulation for a case."""
    S0 = float(init["S0"])
    B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
    M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
    P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}
    if seeding_events is not None:
        return run_ode_with_seeding(params, S0, B0, M0, P0, seeding_events)
    return run_ode(params, S0, B0, M0, P0)


def run_abs_mean_for_case(
    params: ModelParams, 
    init: Dict, 
    replicates: int, 
    seed: int,
    seeding_events: Optional[List[Tuple[float, int, Dict[str, int]]]] = None
) -> Dict:
    """Run ABS simulations and return mean over replicates."""
    rng = np.random.default_rng(seed)
    S0 = int(init["S0"])
    B0 = {int(k): int(v) for k, v in init["B0"].items()}
    M0 = {int(k): int(v) for k, v in init["M0"].items()}
    P0 = {int(k): int(v) for k, v in init["P0"].items()}
    
    abs_acc = None
    for rep in range(replicates):
        if rep % 5 == 0:
            print(f"    ABS replicate {rep+1}/{replicates}...", flush=True)
        roles, rel_ids = init_population_from_counts(S0, B0, M0, P0, params.religions, rng)
        abs_out = run_abs(params, roles, rel_ids, seed=int(rng.integers(1, 10**9)), seeding_events=seeding_events)
        
        if abs_acc is None:
            abs_acc = abs_out
        else:
            # accumulate
            abs_acc["S"] = (np.array(abs_acc["S"]) + np.array(abs_out["S"])).tolist()
            for r in params.religions:
                for key in ["B", "M", "P"]:
                    abs_acc[key][str(r)] = (np.array(abs_acc[key][str(r)]) + np.array(abs_out[key][str(r)])).tolist()
    
    # average
    assert abs_acc is not None
    abs_mean = {
        "t": abs_acc["t"],
        "S": (np.array(abs_acc["S"]) / replicates).tolist(),
        "B": {},
        "M": {},
        "P": {},
    }
    for r in params.religions:
        abs_mean["B"][str(r)] = (np.array(abs_acc["B"][str(r)]) / replicates).tolist()
        abs_mean["M"][str(r)] = (np.array(abs_acc["M"][str(r)]) / replicates).tolist()
        abs_mean["P"][str(r)] = (np.array(abs_acc["P"][str(r)]) / replicates).tolist()
    
    return abs_mean


def main():
    parser = argparse.ArgumentParser(
        description="Generate ODE-ABS comparison summary for 33 cases (11 per scenario type)"
    )
    parser.add_argument("--n_per_type", type=int, default=11, help="Number of cases per scenario type")
    parser.add_argument("--R", type=int, default=30, help="Number of ABS replicates")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="overleaf/data/ode_abs_case_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    
    # Time horizons per scenario (matching paper experiments)
    t_max_by_type = {
        "A": 216.0,  # 216 weeks for Scenario A
        "B": 216.0,  # 216 weeks for Scenario B
        "C": 400.0,  # 400 weeks for Scenario C
    }
    
    # Ensure output directory exists
    out_path = Path(args.out)
    ensure_dirs(out_path.parent)
    
    # Summary rows
    summary_rows: List[Dict[str, str | float]] = []
    
    # Process each scenario type (A, B, C)
    for case_type in ["A", "B", "C"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing Scenario {case_type} ({args.n_per_type} cases)", flush=True)
        print(f"{'=' * 60}", flush=True)
        
        # Get base scenario
        params0, init0 = SCENARIOS[case_type]()
        params0.dt = args.dt
        params0.t_max = t_max_by_type[case_type]
        
        # Seeding events for Scenario B
        seeding_events = None
        if case_type == "B":
            # Scenario B: Use scheduled seeding for strains r=2..6
            # Seeding times: r2=0, r3=16, r4=48, r5=112, r6=167 (weeks)
            seeding_events = [
                (0.0, 2, {"B": 50, "M": 100, "P": 0}),
                (16.0, 3, {"B": 50, "M": 100, "P": 0}),
                (48.0, 4, {"B": 50, "M": 100, "P": 0}),
                (112.0, 5, {"B": 50, "M": 100, "P": 0}),
                (167.0, 6, {"B": 50, "M": 100, "P": 0}),
            ]
        
        # Create RNG for this scenario type (deterministic per type)
        # Use ordinal value of case_type letter (A=1, B=2, C=3) for seed offset
        type_offset = ord(case_type) - ord('A') + 1
        rng = np.random.default_rng(args.seed + type_offset * 1000)
        
        for case_idx in range(args.n_per_type):
            case_id = f"{case_type}{case_idx+1:02d}"
            print(f"\nCase {case_id} starting...", flush=True)
            
            # Sample parameters
            params = sample_params(params0, rng, case_type)
            
            # Run ODE
            print(f"  Solving ODE...", flush=True)
            ode_out = run_ode_for_case(params, init0, seeding_events)
            
            # Run ABS (mean over replicates)
            # Use deterministic seed based on case index
            print(f"  Running ABS ({args.R} replicates)...", flush=True)
            abs_seed = args.seed + type_offset * 1000 + (case_idx + 1) * 100
            abs_mean = run_abs_mean_for_case(params, init0, args.R, abs_seed, seeding_events)
            
            # Compute metrics
            print(f"  Computing metrics...", flush=True)
            # Filter out religion 0 if present
            religions = [r for r in params.religions if r != 0]
            metrics = compare_ode_abs_totals(ode_out, abs_mean, religions)
            
            # Add to summary (only case_type, avg_MAE, avg_R2 as requested)
            summary_rows.append({
                "case_type": case_type,
                "avg_MAE": metrics["mean_mae"],
                "avg_R2": metrics["mean_r2"],
            })
            
            print(f"  Case {case_id} completed! MAE={metrics['mean_mae']:.6f}, R²={metrics['mean_r2']:.4f}", flush=True)
    
    # Write summary CSV
    print(f"\n{'=' * 60}", flush=True)
    print(f"Writing summary to {out_path}...", flush=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_type", "avg_MAE", "avg_R2"])
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Summary saved: {len(summary_rows)} cases", flush=True)
    
    # Print console summary
    print(f"\n{'=' * 60}", flush=True)
    print("Summary Statistics by Case Type:", flush=True)
    print(f"{'=' * 60}", flush=True)
    for case_type in ["A", "B", "C"]:
        type_rows = [r for r in summary_rows if r["case_type"] == case_type]
        if not type_rows:
            continue
        maes = [r["avg_MAE"] for r in type_rows]
        r2s = [r["avg_R2"] for r in type_rows]
        print(f"\nScenario {case_type}:", flush=True)
        print(f"  Count: {len(type_rows)}", flush=True)
        print(f"  avg_MAE - Mean: {np.mean(maes):.6f}, Median: {np.median(maes):.6f}", flush=True)
        print(f"  avg_R2  - Mean: {np.mean(r2s):.4f}, Median: {np.median(r2s):.4f}", flush=True)
    
    print(f"\n{'=' * 60}", flush=True)
    print(f"Total cases: {len(summary_rows)}", flush=True)
    all_maes = [r["avg_MAE"] for r in summary_rows]
    all_r2s = [r["avg_R2"] for r in summary_rows]
    print(f"Overall avg_MAE - Mean: {np.mean(all_maes):.6f}, Median: {np.median(all_maes):.6f}", flush=True)
    print(f"Overall avg_R2  - Mean: {np.mean(all_r2s):.4f}, Median: {np.median(all_r2s):.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Done! CSV saved to: {out_path.absolute()}", flush=True)


if __name__ == "__main__":
    main()
