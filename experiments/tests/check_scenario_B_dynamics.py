# experiments/check_scenario_B_dynamics.py
"""
Automatically validates Scenario B dynamics:
- For each r ∈ {2..6}: peak(y_r) ≥ 0.02, final(y_r) ≤ 0.002
- r1 should remain dominant: final(y_1) > 0.25
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
from src.config import DEFAULT_CONFIG


def compute_metrics(t: np.ndarray, y: np.ndarray) -> dict:
    """Compute peak and final mean (last 20% of time) for a time series."""
    if len(y) == 0:
        return {"peak": 0.0, "final_mean": 0.0}
    
    peak = float(np.max(y))
    
    # Final mean: average over last 20% of time
    final_start_idx = int(0.8 * len(y))
    final_mean = float(np.mean(y[final_start_idx:])) if final_start_idx < len(y) else float(y[-1])
    
    return {"peak": peak, "final_mean": final_mean}


def check_scenario_B_dynamics():
    """Load Scenario B results and check dynamics."""
    cfg = DEFAULT_CONFIG
    json_path = cfg.runs_dir / "scenario_B.json"
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run Scenario B first!")
        return
    
    print("=" * 80)
    print("Scenario B Dynamics Check")
    print("=" * 80)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Use ABS mean results
    abs_mean = data["abs_mean"]
    t = np.array(abs_mean["t"], dtype=float)
    S = np.array(abs_mean["S"], dtype=float)
    
    # Compute total population N at each time step
    N = S.copy()
    religions = sorted([int(k) for k in abs_mean["B"].keys()])
    
    for r in religions:
        N += np.array(abs_mean["B"][str(r)], dtype=float)
        N += np.array(abs_mean["M"][str(r)], dtype=float)
        N += np.array(abs_mean["P"][str(r)], dtype=float)
    
    # Compute shares for each religion
    shares = {}
    for r in religions:
        Br = np.array(abs_mean["B"][str(r)], dtype=float)
        Mr = np.array(abs_mean["M"][str(r)], dtype=float)
        Pr = np.array(abs_mean["P"][str(r)], dtype=float)
        shares[r] = (Br + Mr + Pr) / np.maximum(N, 1e-12)
    
    print("\nTarget Criteria:")
    print("  - For r ∈ {2..6}: peak ≥ 0.02, final ≤ 0.002")
    print("  - For r=1: final > 0.25 (dominant)")
    print("\n" + "-" * 80)
    
    # Check r=1 dominance
    r1_metrics = compute_metrics(t, shares[1])
    r1_final = r1_metrics["final_mean"]
    
    print(f"\nr=1 (dominant strain):")
    print(f"  Final share: {r1_final:.4f}")
    if r1_final > 0.25:
        print(f"  OK: r=1 remains dominant (final > 0.25)")
    else:
        print(f"  WARNING: r=1 may not be dominant (final ≤ 0.25)")
    
    # Check r=2..6 transient dynamics
    print(f"\nr=2..6 (transient cults):")
    all_ok = True
    
    for r in sorted([r for r in religions if r >= 2]):
        metrics = compute_metrics(t, shares[r])
        peak = metrics["peak"]
        final = metrics["final_mean"]
        
        peak_ok = peak >= 0.02
        final_ok = final <= 0.002
        
        status_peak = "OK" if peak_ok else "FAIL"
        status_final = "OK" if final_ok else "FAIL"
        
        print(f"\n  r={r}:")
        print(f"    Peak: {peak:.4f} (target: ≥ 0.02) [{status_peak}]")
        print(f"    Final: {final:.4f} (target: ≤ 0.002) [{status_final}]")
        
        if not (peak_ok and final_ok):
            all_ok = False
    
    print("\n" + "-" * 80)
    if all_ok and r1_final > 0.25:
        print("\nSUCCESS: All criteria met!")
    else:
        print("\nWARNING: Some criteria not met. Consider parameter tuning.")
    
    print("=" * 80)


if __name__ == "__main__":
    check_scenario_B_dynamics()
