#!/usr/bin/env python3
"""Complete verification of table against scenarios.py"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import scenario_A, scenario_B, scenario_C

pa, ia = scenario_A()
pb, ib = scenario_B()
pc, ic = scenario_C()

print("=" * 80)
print("COMPLETE TABLE VERIFICATION")
print("=" * 80)

errors = []
warnings = []

# ============================================================================
# SCENARIO A
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO A")
print("=" * 80)

# Demography
if pa.b != 0.01:
    errors.append("A: b should be 0.01")
if pa.mu != 0.01:
    errors.append("A: mu should be 0.01")

# Religion 1
if abs(pa.beta0[1] - 0.03) > 0.001:
    errors.append(f"A: beta0[1] should be 0.03, got {pa.beta0[1]}")
if abs(pa.q[1] - 0.05) > 0.001:
    errors.append(f"A: q[1] should be 0.05, got {pa.q[1]}")
if abs(pa.sigma[1] - 0.02) > 0.001:
    errors.append(f"A: sigma[1] should be 0.02, got {pa.sigma[1]}")
if abs(pa.kappa[1] - 0.01) > 0.001:
    errors.append(f"A: kappa[1] should be 0.01, got {pa.kappa[1]}")
if abs(pa.tauB[1] - 0.002) > 0.001:
    errors.append(f"A: tauB[1] should be 0.002, got {pa.tauB[1]}")
if abs(pa.tauM[1] - 0.003) > 0.001:
    errors.append(f"A: tauM[1] should be 0.003, got {pa.tauM[1]}")
if abs(pa.rhoB[1] - 0.005) > 0.001:
    errors.append(f"A: rhoB[1] should be 0.005, got {pa.rhoB[1]}")
if abs(pa.rhoM[1] - 0.004) > 0.001:
    errors.append(f"A: rhoM[1] should be 0.004, got {pa.rhoM[1]}")
if abs(pa.rhoP[1] - 0.001) > 0.001:
    errors.append(f"A: rhoP[1] should be 0.001, got {pa.rhoP[1]}")

# Religion 2
if abs(pa.beta0[2] - 0.45) > 0.001:
    errors.append(f"A: beta0[2] should be 0.45, got {pa.beta0[2]}")
if abs(pa.q[2] - 0.1) > 0.001:
    errors.append(f"A: q[2] should be 0.1, got {pa.q[2]}")
if abs(pa.sigma[2] - 0.03) > 0.001:
    errors.append(f"A: sigma[2] should be 0.03, got {pa.sigma[2]}")
if abs(pa.kappa[2] - 0.01) > 0.001:
    errors.append(f"A: kappa[2] should be 0.01, got {pa.kappa[2]}")
if abs(pa.tauB[2] - 0.003) > 0.001:
    errors.append(f"A: tauB[2] should be 0.003, got {pa.tauB[2]}")
if abs(pa.tauM[2] - 0.004) > 0.001:
    errors.append(f"A: tauM[2] should be 0.004, got {pa.tauM[2]}")
if abs(pa.rhoB[2] - 0.004) > 0.001:
    errors.append(f"A: rhoB[2] should be 0.004, got {pa.rhoB[2]}")
if abs(pa.rhoM[2] - 0.003) > 0.001:
    errors.append(f"A: rhoM[2] should be 0.003, got {pa.rhoM[2]}")
if abs(pa.rhoP[2] - 0.001) > 0.001:
    errors.append(f"A: rhoP[2] should be 0.001, got {pa.rhoP[2]}")

# Mutation
if pa.nu[1].get(2, 0.0) != 0.002:
    errors.append(f"A: nu[1][2] should be 0.002, got {pa.nu[1].get(2, 0.0)}")
if pa.nu[2].get(1, 0.0) != 0.0:
    errors.append(f"A: nu[2][1] should be 0.0, got {pa.nu[2].get(1, 0.0)}")

# Initial conditions
if ia['S0'] != 6000:
    errors.append(f"A: S0 should be 6000, got {ia['S0']}")
if ia['B0'][1] != 3000:
    errors.append(f"A: B0[1] should be 3000, got {ia['B0'][1]}")
if ia['M0'][1] != 200:
    errors.append(f"A: M0[1] should be 200, got {ia['M0'][1]}")
if ia['P0'][1] != 800:
    errors.append(f"A: P0[1] should be 800, got {ia['P0'][1]}")
if ia['B0'][2] != 0:
    errors.append(f"A: B0[2] should be 0, got {ia['B0'][2]}")
if ia['M0'][2] != 0:
    errors.append(f"A: M0[2] should be 0, got {ia['M0'][2]}")
if ia['P0'][2] != 0:
    errors.append(f"A: P0[2] should be 0, got {ia['P0'][2]}")

# Simulation
if abs(pa.dt - 0.1) > 0.001:
    errors.append(f"A: dt should be 0.1, got {pa.dt}")
if pa.t_max != 200.0:
    warnings.append(f"A: t_max in code is {pa.t_max}, but run_scenarios.py overrides to 216.0")

# ============================================================================
# SCENARIO B
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO B")
print("=" * 80)

# Demography
if pb.b != 0.01:
    errors.append("B: b should be 0.01")
if pb.mu != 0.01:
    errors.append("B: mu should be 0.01")

# Religion 1
if abs(pb.beta0[1] - 0.04) > 0.001:
    errors.append(f"B: beta0[1] should be 0.04, got {pb.beta0[1]}")
if abs(pb.q[1] - 0.05) > 0.001:
    errors.append(f"B: q[1] should be 0.05, got {pb.q[1]}")
if abs(pb.sigma[1] - 0.02) > 0.001:
    errors.append(f"B: sigma[1] should be 0.02, got {pb.sigma[1]}")
if abs(pb.kappa[1] - 0.02) > 0.001:
    errors.append(f"B: kappa[1] should be 0.02, got {pb.kappa[1]}")
if abs(pb.tauB[1] - 0.002) > 0.001:
    errors.append(f"B: tauB[1] should be 0.002, got {pb.tauB[1]}")
if abs(pb.tauM[1] - 0.003) > 0.001:
    errors.append(f"B: tauM[1] should be 0.003, got {pb.tauM[1]}")
if abs(pb.rhoB[1] - 0.006) > 0.001:
    errors.append(f"B: rhoB[1] should be 0.006, got {pb.rhoB[1]}")
if abs(pb.rhoM[1] - 0.006) > 0.001:
    errors.append(f"B: rhoM[1] should be 0.006, got {pb.rhoM[1]}")
if abs(pb.rhoP[1] - 0.001) > 0.001:
    errors.append(f"B: rhoP[1] should be 0.001, got {pb.rhoP[1]}")

# Religion 2
if abs(pb.beta0[2] - 0.02) > 0.001:
    errors.append(f"B: beta0[2] should be 0.02, got {pb.beta0[2]}")
if abs(pb.q[2] - 0.02) > 0.001:
    errors.append(f"B: q[2] should be 0.02, got {pb.q[2]}")
if abs(pb.sigma[2] - 0.005) > 0.001:
    errors.append(f"B: sigma[2] should be 0.005, got {pb.sigma[2]}")
if abs(pb.kappa[2] - 0.02) > 0.001:
    errors.append(f"B: kappa[2] should be 0.02, got {pb.kappa[2]}")
if abs(pb.tauB[2] - 0.0003) > 0.0001:
    errors.append(f"B: tauB[2] should be 0.0003, got {pb.tauB[2]}")
if abs(pb.tauM[2] - 0.0003) > 0.0001:
    errors.append(f"B: tauM[2] should be 0.0003, got {pb.tauM[2]}")
if abs(pb.rhoB[2] - 0.05) > 0.001:
    errors.append(f"B: rhoB[2] should be 0.05, got {pb.rhoB[2]}")
if abs(pb.rhoM[2] - 0.15) > 0.001:
    errors.append(f"B: rhoM[2] should be 0.15, got {pb.rhoM[2]}")
if abs(pb.rhoP[2] - 0.4) > 0.001:
    errors.append(f"B: rhoP[2] should be 0.4, got {pb.rhoP[2]}")

# Religions 3-6 (should be same as 2)
for r in [3, 4, 5, 6]:
    if abs(pb.beta0[r] - 0.02) > 0.001:
        errors.append(f"B: beta0[{r}] should be 0.02, got {pb.beta0[r]}")
    if abs(pb.q[r] - 0.02) > 0.001:
        errors.append(f"B: q[{r}] should be 0.02, got {pb.q[r]}")
    if abs(pb.sigma[r] - 0.005) > 0.001:
        errors.append(f"B: sigma[{r}] should be 0.005, got {pb.sigma[r]}")
    if abs(pb.kappa[r] - 0.02) > 0.001:
        errors.append(f"B: kappa[{r}] should be 0.02, got {pb.kappa[r]}")
    if abs(pb.tauB[r] - 0.0003) > 0.0001:
        errors.append(f"B: tauB[{r}] should be 0.0003, got {pb.tauB[r]}")
    if abs(pb.tauM[r] - 0.0003) > 0.0001:
        errors.append(f"B: tauM[{r}] should be 0.0003, got {pb.tauM[r]}")
    if abs(pb.rhoB[r] - 0.05) > 0.001:
        errors.append(f"B: rhoB[{r}] should be 0.05, got {pb.rhoB[r]}")
    if abs(pb.rhoM[r] - 0.15) > 0.001:
        errors.append(f"B: rhoM[{r}] should be 0.15, got {pb.rhoM[r]}")
    if abs(pb.rhoP[r] - 0.4) > 0.001:
        errors.append(f"B: rhoP[{r}] should be 0.4, got {pb.rhoP[r]}")

# Mutation (all should be 0)
for r in [1, 2, 3, 4, 5, 6]:
    for s in [1, 2, 3, 4, 5, 6]:
        if r != s:
            if pb.nu[r].get(s, 0.0) != 0.0:
                errors.append(f"B: nu[{r}][{s}] should be 0.0, got {pb.nu[r].get(s, 0.0)}")

# Initial conditions
if ib['S0'] != 5000:
    errors.append(f"B: S0 should be 5000, got {ib['S0']}")
if ib['B0'][1] != 4200:
    errors.append(f"B: B0[1] should be 4200, got {ib['B0'][1]}")
if ib['M0'][1] != 300:
    errors.append(f"B: M0[1] should be 300, got {ib['M0'][1]}")
if ib['P0'][1] != 500:
    errors.append(f"B: P0[1] should be 500, got {ib['P0'][1]}")
for r in [2, 3, 4, 5, 6]:
    if ib['B0'][r] != 0:
        errors.append(f"B: B0[{r}] should be 0, got {ib['B0'][r]}")
    if ib['M0'][r] != 0:
        errors.append(f"B: M0[{r}] should be 0, got {ib['M0'][r]}")
    if ib['P0'][r] != 0:
        errors.append(f"B: P0[{r}] should be 0, got {ib['P0'][r]}")

# Simulation
if abs(pb.dt - 0.1) > 0.001:
    errors.append(f"B: dt should be 0.1, got {pb.dt}")
if pb.t_max != 216.0:
    warnings.append(f"B: t_max in code is {pb.t_max}, but run_scenarios.py overrides to 216.0")

# ============================================================================
# SCENARIO C
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO C")
print("=" * 80)

# Demography
if pc.b != 0.01:
    errors.append("C: b should be 0.01")
if pc.mu != 0.01:
    errors.append("C: mu should be 0.01")

# Religion 1
if abs(pc.beta0[1] - 0.28) > 0.001:
    errors.append(f"C: beta0[1] should be 0.28, got {pc.beta0[1]}")
if abs(pc.q[1] - 0.05) > 0.001:
    errors.append(f"C: q[1] should be 0.05, got {pc.q[1]}")
if abs(pc.sigma[1] - 0.018) > 0.001:
    errors.append(f"C: sigma[1] should be 0.018, got {pc.sigma[1]}")
if abs(pc.kappa[1] - 0.018) > 0.001:
    errors.append(f"C: kappa[1] should be 0.018, got {pc.kappa[1]}")
if abs(pc.tauB[1] - 0.0015) > 0.001:
    errors.append(f"C: tauB[1] should be 0.0015, got {pc.tauB[1]}")
if abs(pc.tauM[1] - 0.002) > 0.001:
    errors.append(f"C: tauM[1] should be 0.002, got {pc.tauM[1]}")
if abs(pc.rhoB[1] - 0.006) > 0.001:
    errors.append(f"C: rhoB[1] should be 0.006, got {pc.rhoB[1]}")
if abs(pc.rhoM[1] - 0.006) > 0.001:
    errors.append(f"C: rhoM[1] should be 0.006, got {pc.rhoM[1]}")
if abs(pc.rhoP[1] - 0.001) > 0.001:
    errors.append(f"C: rhoP[1] should be 0.001, got {pc.rhoP[1]}")

# Religion 2
if abs(pc.beta0[2] - 0.55) > 0.001:
    errors.append(f"C: beta0[2] should be 0.55, got {pc.beta0[2]}")
if abs(pc.q[2] - 0.05) > 0.001:
    errors.append(f"C: q[2] should be 0.05, got {pc.q[2]}")
if abs(pc.sigma[2] - 0.018) > 0.001:
    errors.append(f"C: sigma[2] should be 0.018, got {pc.sigma[2]}")
if abs(pc.kappa[2] - 0.018) > 0.001:
    errors.append(f"C: kappa[2] should be 0.018, got {pc.kappa[2]}")
if abs(pc.tauB[2] - 0.0015) > 0.001:
    errors.append(f"C: tauB[2] should be 0.0015, got {pc.tauB[2]}")
if abs(pc.tauM[2] - 0.002) > 0.001:
    errors.append(f"C: tauM[2] should be 0.002, got {pc.tauM[2]}")
if abs(pc.rhoB[2] - 0.0225) > 0.001:
    errors.append(f"C: rhoB[2] should be 0.0225, got {pc.rhoB[2]}")
if abs(pc.rhoM[2] - 0.03) > 0.001:
    errors.append(f"C: rhoM[2] should be 0.03, got {pc.rhoM[2]}")
if abs(pc.rhoP[2] - 0.003) > 0.001:
    errors.append(f"C: rhoP[2] should be 0.003, got {pc.rhoP[2]}")

# Mutation
if pc.nu[1].get(2, 0.0) != 0.0:
    errors.append(f"C: nu[1][2] should be 0.0, got {pc.nu[1].get(2, 0.0)}")
if pc.nu[2].get(1, 0.0) != 0.0:
    errors.append(f"C: nu[2][1] should be 0.0, got {pc.nu[2].get(1, 0.0)}")

# Initial conditions
if ic['S0'] != 6500:
    errors.append(f"C: S0 should be 6500, got {ic['S0']}")
if ic['B0'][1] != 2000:
    errors.append(f"C: B0[1] should be 2000, got {ic['B0'][1]}")
if ic['M0'][1] != 120:
    errors.append(f"C: M0[1] should be 120, got {ic['M0'][1]}")
if ic['P0'][1] != 180:
    errors.append(f"C: P0[1] should be 180, got {ic['P0'][1]}")
if ic['B0'][2] != 100:
    errors.append(f"C: B0[2] should be 100, got {ic['B0'][2]}")
if ic['M0'][2] != 0:
    errors.append(f"C: M0[2] should be 0, got {ic['M0'][2]}")
if ic['P0'][2] != 0:
    errors.append(f"C: P0[2] should be 0, got {ic['P0'][2]}")

# Simulation
if abs(pc.dt - 0.1) > 0.001:
    errors.append(f"C: dt should be 0.1, got {pc.dt}")
if pc.t_max != 400.0:
    errors.append(f"C: t_max should be 400.0, got {pc.t_max}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if errors:
    print(f"\nERRORS FOUND: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
else:
    print("\nNo errors found!")

if warnings:
    print(f"\nWARNINGS: {len(warnings)}")
    for w in warnings:
        print(f"  - {w}")

print("\n" + "=" * 80)
