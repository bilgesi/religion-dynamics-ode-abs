#!/usr/bin/env python3
"""Check table errors - simple version."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import scenario_A, scenario_B, scenario_C

pa, ia = scenario_A()
pb, ib = scenario_B()
pc, ic = scenario_C()

errors = []

# Scenario A
if abs(pa.beta0[1] - 0.03) > 0.001:
    errors.append(f"A: beta0[1] should be 0.03, table shows 0.3")
if pa.t_max != 200.0:
    errors.append(f"A: t_max in code is {pa.t_max}, but run_scenarios.py overrides to 216.0")

# Scenario B
if abs(pb.beta0[1] - 0.04) > 0.001:
    errors.append(f"B: beta0[1] should be 0.04, table shows 0.3")
if abs(pb.beta0[2] - 0.02) > 0.001:
    errors.append(f"B: beta0[2] should be 0.02, table shows 0.18")
if abs(pb.q[2] - 0.02) > 0.001:
    errors.append(f"B: q[2] should be 0.02, table shows 0.06")
if abs(pb.sigma[2] - 0.005) > 0.001:
    errors.append(f"B: sigma[2] should be 0.005, table shows 0.02")
if abs(pb.kappa[2] - 0.02) > 0.001:
    errors.append(f"B: kappa[2] should be 0.02, table shows 0.04")
if abs(pb.tauB[2] - 0.0003) > 0.0001:
    errors.append(f"B: tauB[2] should be 0.0003, table shows 0.0005")
if abs(pb.tauM[2] - 0.0003) > 0.0001:
    errors.append(f"B: tauM[2] should be 0.0003, table shows 0.0005")
if abs(pb.rhoB[2] - 0.05) > 0.001:
    errors.append(f"B: rhoB[2] should be 0.05, table shows 0.02")
if abs(pb.rhoM[2] - 0.15) > 0.001:
    errors.append(f"B: rhoM[2] should be 0.15, table shows 0.02")
if abs(pb.rhoP[2] - 0.4) > 0.001:
    errors.append(f"B: rhoP[2] should be 0.4, table shows 0.01")
if pb.nu[1].get(2, 0.0) != 0.0:
    errors.append(f"B: nu[1][2] should be 0.0 (mutation disabled), table shows 0.005")

# Scenario C
if abs(pc.beta0[2] - 0.55) > 0.001:
    errors.append(f"C: beta0[2] should be 0.55, table shows 0.26")
if abs(pc.rhoB[2] - 0.0225) > 0.001:
    errors.append(f"C: rhoB[2] should be 0.0225, table shows 0.006")
if abs(pc.rhoM[2] - 0.03) > 0.001:
    errors.append(f"C: rhoM[2] should be 0.03, table shows 0.006")
if abs(pc.rhoP[2] - 0.003) > 0.001:
    errors.append(f"C: rhoP[2] should be 0.003, table shows 0.001")
if pc.nu[1].get(2, 0.0) != 0.0:
    errors.append(f"C: nu[1][2] should be 0.0, table shows 0.0002")
if pc.nu[2].get(1, 0.0) != 0.0:
    errors.append(f"C: nu[2][1] should be 0.0, table shows 0.0002")

print("=" * 70)
print("TABLE ERRORS FOUND:")
print("=" * 70)
if errors:
    for e in errors:
        print(f"  - {e}")
else:
    print("  No errors found!")
print("=" * 70)
