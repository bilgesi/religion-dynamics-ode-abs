#!/usr/bin/env python3
"""
Compute quantitative milestones for scenarios A, B, C from replicate-mean time series.
"""

import json
from pathlib import Path
import numpy as np

def compute_share(t, S, B, M, P, religions):
    """Compute population shares for each religion."""
    N = np.array(S, dtype=float)
    for r in religions:
        r_str = str(r)
        if r_str in B and r_str in M and r_str in P:
            N = N + np.array(B[r_str], dtype=float) + np.array(M[r_str], dtype=float) + np.array(P[r_str], dtype=float)
    
    shares = {}
    for r in religions:
        r_str = str(r)
        if r_str in B and r_str in M and r_str in P:
            shares[r] = (np.array(B[r_str], dtype=float) + np.array(M[r_str], dtype=float) + np.array(P[r_str], dtype=float)) / np.maximum(N, 1e-12)
        else:
            shares[r] = np.zeros_like(t, dtype=float)
    return shares, N

def scenario_A_milestones(data):
    """Panel (a): takeoff time when y2 crosses 0.01, crossover time when y2=y1, final shares at T."""
    ode = data["ode"]
    
    t = np.array(ode["t"], dtype=float)
    S = np.array(ode["S"], dtype=float)
    B = {r: np.array(ode["B"][str(r)], dtype=float) for r in [1, 2]}
    M = {r: np.array(ode["M"][str(r)], dtype=float) for r in [1, 2]}
    P = {r: np.array(ode["P"][str(r)], dtype=float) for r in [1, 2]}
    
    # Compute shares directly
    N = S + B[1] + M[1] + P[1] + B[2] + M[2] + P[2]
    y1 = (B[1] + M[1] + P[1]) / np.maximum(N, 1e-12)
    y2 = (B[2] + M[2] + P[2]) / np.maximum(N, 1e-12)
    
    # Takeoff: y2 crosses 0.01
    takeoff_idx = np.where(y2 >= 0.01)[0]
    takeoff_time = float(t[takeoff_idx[0]]) if len(takeoff_idx) > 0 else None
    
    # Crossover: y2 = y1 (find where y2 - y1 crosses zero)
    diff = y2 - y1
    crossover_idx = np.where(diff >= 0)[0]
    crossover_time = float(t[crossover_idx[0]]) if len(crossover_idx) > 0 else None
    
    # Final shares at T
    T = float(t[-1])
    y1_final = float(y1[-1])
    y2_final = float(y2[-1])
    
    return {
        "takeoff_time": takeoff_time,
        "crossover_time": crossover_time,
        "T": T,
        "y1_final": y1_final,
        "y2_final": y2_final
    }

def scenario_B_milestones(data):
    """Panel (b): number of bursts (peak count) for minor strains, max peak share among minors, final total share of minors at T."""
    ode = data["ode"]
    
    t = np.array(ode["t"], dtype=float)
    S = np.array(ode["S"], dtype=float)
    religions = [1, 2, 3, 4, 5, 6]
    B = {r: np.array(ode["B"][str(r)], dtype=float) for r in religions}
    M = {r: np.array(ode["M"][str(r)], dtype=float) for r in religions}
    P = {r: np.array(ode["P"][str(r)], dtype=float) for r in religions}
    
    # Compute shares directly
    N = S.copy()
    for r in religions:
        N = N + B[r] + M[r] + P[r]
    shares = {r: (B[r] + M[r] + P[r]) / np.maximum(N, 1e-12) for r in religions}
    
    # Minor strains: r=2..6
    minor_strains = [2, 3, 4, 5, 6]
    peaks = []
    max_peak = 0.0
    
    for r in minor_strains:
        y = shares[r]
        peak_idx = np.argmax(y)
        peak_val = float(y[peak_idx])
        peaks.append(peak_val)
        if peak_val > max_peak:
            max_peak = peak_val
    
    num_bursts = len(minor_strains)  # Each strain has one peak
    
    # Final total share of minors
    T = float(t[-1])
    minors_final = sum(float(shares[r][-1]) for r in minor_strains)
    
    return {
        "num_bursts": num_bursts,
        "max_peak_share": max_peak,
        "T": T,
        "minors_final_total_share": minors_final
    }

def scenario_C_milestones(data):
    """Panel (c): equilibrium shares y1*, y2* (mean over last 20% of horizon), and settling time to stay within ±0.01."""
    ode = data["ode"]
    
    t = np.array(ode["t"], dtype=float)
    S = np.array(ode["S"], dtype=float)
    B = {r: np.array(ode["B"][str(r)], dtype=float) for r in [1, 2]}
    M = {r: np.array(ode["M"][str(r)], dtype=float) for r in [1, 2]}
    P = {r: np.array(ode["P"][str(r)], dtype=float) for r in [1, 2]}
    
    # Compute shares directly
    N = S + B[1] + M[1] + P[1] + B[2] + M[2] + P[2]
    y1 = (B[1] + M[1] + P[1]) / np.maximum(N, 1e-12)
    y2 = (B[2] + M[2] + P[2]) / np.maximum(N, 1e-12)
    
    # Equilibrium: mean over last 20% of horizon
    T = float(t[-1])
    last_20_start = int(len(t) * 0.8)
    y1_eq = float(np.mean(y1[last_20_start:]))
    y2_eq = float(np.mean(y2[last_20_start:]))
    
    # Settling time: time when both stay within ±0.01 of equilibrium
    tolerance = 0.01
    settling_idx = None
    for i in range(len(t)):
        if (abs(y1[i] - y1_eq) <= tolerance and 
            abs(y2[i] - y2_eq) <= tolerance):
            # Check if it stays within tolerance for rest of time
            remaining = y1[i:] - y1_eq
            remaining2 = y2[i:] - y2_eq
            if (np.all(np.abs(remaining) <= tolerance) and 
                np.all(np.abs(remaining2) <= tolerance)):
                settling_idx = i
                break
    
    settling_time = float(t[settling_idx]) if settling_idx is not None else None
    
    return {
        "y1_equilibrium": y1_eq,
        "y2_equilibrium": y2_eq,
        "T": T,
        "settling_time": settling_time
    }

def main():
    base_dir = Path(".")
    runs_dir = base_dir / "outputs" / "runs"
    
    results = {}
    
    for scenario in ["A", "B", "C"]:
        json_path = runs_dir / f"scenario_{scenario}.json"
        if not json_path.exists():
            print(f"WARNING: {json_path} not found, skipping scenario {scenario}")
            continue
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if scenario == "A":
            results["A"] = scenario_A_milestones(data)
        elif scenario == "B":
            results["B"] = scenario_B_milestones(data)
        elif scenario == "C":
            results["C"] = scenario_C_milestones(data)
    
    # Print results
    print("=" * 70)
    print("SCENARIO MILESTONES")
    print("=" * 70)
    
    if "A" in results:
        print("\nScenario A:")
        r = results["A"]
        print(f"  Takeoff time (y2 >= 0.01): {r['takeoff_time']:.1f}" if r['takeoff_time'] else "  Takeoff time: N/A")
        print(f"  Crossover time (y2 = y1): {r['crossover_time']:.1f}" if r['crossover_time'] else "  Crossover time: N/A")
        print(f"  Final shares at T={r['T']:.1f}: y1={r['y1_final']:.3f}, y2={r['y2_final']:.3f}")
    
    if "B" in results:
        print("\nScenario B:")
        r = results["B"]
        print(f"  Number of bursts (minor strains): {r['num_bursts']}")
        print(f"  Max peak share among minors: {r['max_peak_share']:.3f}")
        print(f"  Final total share of minors at T={r['T']:.1f}: {r['minors_final_total_share']:.3f}")
    
    if "C" in results:
        print("\nScenario C:")
        r = results["C"]
        print(f"  Equilibrium shares (last 20%): y1*={r['y1_equilibrium']:.3f}, y2*={r['y2_equilibrium']:.3f}")
        print(f"  Settling time (within ±0.01): {r['settling_time']:.1f}" if r['settling_time'] else "  Settling time: N/A")
        print(f"  Horizon T: {r['T']:.1f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
