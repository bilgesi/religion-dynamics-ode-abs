"""
Check Scenario B output results from JSON files.

Loads saved scenario data and prints summary statistics.
Internal validation script for debugging.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np

# Load results
with open("outputs/runs/scenario_B.json", "r") as f:
    data = json.load(f)

ode = data["ode"]
t = np.array(ode["t"], dtype=float)
S = np.array(ode["S"], dtype=float)

# Compute total population
N = S.copy()
for r in [1, 2, 3, 4, 5, 6]:
    N += np.array(ode["B"][str(r)], dtype=float)
    N += np.array(ode["M"][str(r)], dtype=float)
    N += np.array(ode["P"][str(r)], dtype=float)

print("=" * 80)
print("Scenario B ODE Results")
print("=" * 80)
print(f"Time range: {t[0]:.1f} - {t[-1]:.1f} weeks")
print(f"Total population at start: {N[0]:.0f}")
print(f"Total population at end: {N[-1]:.0f}")
print()

for r in [1, 2, 3, 4, 5, 6]:
    B_r = np.array(ode["B"][str(r)], dtype=float)
    M_r = np.array(ode["M"][str(r)], dtype=float)
    P_r = np.array(ode["P"][str(r)], dtype=float)
    y_r = (B_r + M_r + P_r) / np.maximum(N, 1e-12)
    
    peak = np.max(y_r)
    peak_idx = np.argmax(y_r)
    peak_time = t[peak_idx]
    
    final = np.mean(y_r[int(len(t) * 0.8) :])
    
    if r == 1:
        print(f"r={r} (dominant):")
    else:
        print(f"r={r} (transient cult):")
    print(f"  Peak: {peak:.4f} at week {peak_time:.1f}")
    print(f"  Final (last 20%): {final:.4f}")
    print(f"  Initial: {y_r[0]:.4f}")
    print()

# Check if r=1 remains dominant
y1 = (np.array(ode["B"]["1"], dtype=float) + np.array(ode["M"]["1"], dtype=float) + np.array(ode["P"]["1"], dtype=float)) / np.maximum(N, 1e-12)
y2 = (np.array(ode["B"]["2"], dtype=float) + np.array(ode["M"]["2"], dtype=float) + np.array(ode["P"]["2"], dtype=float)) / np.maximum(N, 1e-12)
final1 = np.mean(y1[int(len(t) * 0.8) :])
final2 = np.mean(y2[int(len(t) * 0.8) :])

print("=" * 80)
print("Dominance Check:")
print(f"  r=1 final: {final1:.4f}")
print(f"  r=2 final: {final2:.4f}")
if final1 > final2:
    print("  OK: r=1 remains dominant")
else:
    print("  WARNING: r=2 takes over!")
print("=" * 80)
