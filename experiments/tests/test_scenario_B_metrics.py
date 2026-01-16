"""
Test Scenario B and compute per-strain metrics.

Validates that each strain meets peak and final share criteria.
Internal validation script for debugging.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scenarios import scenario_B
from src.ode.solve_ode_with_seeding import run_ode_with_seeding
import numpy as np

params, init = scenario_B()
S0 = float(init["S0"])
B0 = {r: float(init["B0"].get(r, 0)) for r in params.religions}
M0 = {r: float(init["M0"].get(r, 0)) for r in params.religions}
P0 = {r: float(init["P0"].get(r, 0)) for r in params.religions}

seeding = [
    (0.0, 2, {"B": 180, "M": 25, "P": 0}),
    (16.0, 3, {"B": 180, "M": 25, "P": 0}),
    (48.0, 4, {"B": 180, "M": 25, "P": 0}),
    (112.0, 5, {"B": 180, "M": 25, "P": 0}),
    (167.0, 6, {"B": 180, "M": 25, "P": 0}),
]

print("Running Scenario B ODE...")
out = run_ode_with_seeding(params, S0, B0, M0, P0, seeding)
t = np.array(out["t"])
S = np.array(out["S"])
N = S.copy()
for r in params.religions:
    N += np.array(out["B"][str(r)]) + np.array(out["M"][str(r)]) + np.array(out["P"][str(r)])

print("\n" + "=" * 80)
print("Scenario B Metrics (peak >= 0.02, final <= 0.001)")
print("=" * 80)
all_ok = True
for r in [2, 3, 4, 5, 6]:
    y_r = (np.array(out["B"][str(r)]) + np.array(out["M"][str(r)]) + np.array(out["P"][str(r)])) / np.maximum(N, 1e-12)
    peak = np.max(y_r)
    final = np.mean(y_r[int(len(t) * 0.8) :])
    peak_ok = peak >= 0.02
    final_ok = final <= 0.001
    status = "OK" if (peak_ok and final_ok) else "FAIL"
    if not (peak_ok and final_ok):
        all_ok = False
    print(f"r={r}: peak={peak:.4f} {'OK' if peak_ok else 'FAIL'}, final={final:.4f} {'OK' if final_ok else 'FAIL'} [{status}]")

# Check if r=2 takes over
y1 = (np.array(out["B"]["1"]) + np.array(out["M"]["1"]) + np.array(out["P"]["1"])) / np.maximum(N, 1e-12)
y2 = (np.array(out["B"]["2"]) + np.array(out["M"]["2"]) + np.array(out["P"]["2"])) / np.maximum(N, 1e-12)
final1 = np.mean(y1[int(len(t) * 0.8) :])
final2 = np.mean(y2[int(len(t) * 0.8) :])
print(f"\nr=1 final: {final1:.4f}, r=2 final: {final2:.4f}")
if final2 > final1:
    print("WARNING: r=2 takes over! Need to reduce beta0[2] or increase rhoM[2]/rhoB[2]")
    all_ok = False
else:
    print("OK: r=1 remains dominant")

print("=" * 80)
if all_ok:
    print("ALL CHECKS PASSED!")
else:
    print("SOME CHECKS FAILED - need parameter adjustment")
