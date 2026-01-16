#!/usr/bin/env python3
"""Verify table values against actual scenario parameters."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import scenario_A, scenario_B, scenario_C

pa, ia = scenario_A()
pb, ib = scenario_B()
pc, ic = scenario_C()

print("=" * 70)
print("VERIFICATION: Table vs Actual Parameters")
print("=" * 70)

# Scenario A
print("\nSCENARIO A:")
print(f"  beta0[1]: Table=0.3, Actual={pa.beta0[1]:.3f} {'OK' if abs(pa.beta0[1] - 0.3) < 0.01 else 'WRONG!'}")
print(f"  beta0[2]: Table=0.45, Actual={pa.beta0[2]:.3f} {'OK' if abs(pa.beta0[2] - 0.45) < 0.01 else 'WRONG!'}")
print(f"  t_max: Table=200, Actual={pa.t_max:.1f} (but run_scenarios.py overrides to 216.0)")

# Scenario B
print("\nSCENARIO B:")
print(f"  beta0[1]: Table=0.3, Actual={pb.beta0[1]:.3f} {'✓' if abs(pb.beta0[1] - 0.3) < 0.01 else '✗ WRONG!'}")
print(f"  beta0[2]: Table=0.18, Actual={pb.beta0[2]:.3f} {'✓' if abs(pb.beta0[2] - 0.18) < 0.01 else '✗ WRONG!'}")
print(f"  q[2]: Table=0.06, Actual={pb.q[2]:.3f} {'✓' if abs(pb.q[2] - 0.06) < 0.01 else '✗ WRONG!'}")
print(f"  sigma[2]: Table=0.02, Actual={pb.sigma[2]:.3f} {'✓' if abs(pb.sigma[2] - 0.02) < 0.01 else '✗ WRONG!'}")
print(f"  kappa[2]: Table=0.04, Actual={pb.kappa[2]:.3f} {'✓' if abs(pb.kappa[2] - 0.04) < 0.01 else '✗ WRONG!'}")
print(f"  tauB[2]: Table=0.0005, Actual={pb.tauB[2]:.4f} {'✓' if abs(pb.tauB[2] - 0.0005) < 0.0001 else '✗ WRONG!'}")
print(f"  tauM[2]: Table=0.0005, Actual={pb.tauM[2]:.4f} {'✓' if abs(pb.tauM[2] - 0.0005) < 0.0001 else '✗ WRONG!'}")
print(f"  rhoB[2]: Table=0.02, Actual={pb.rhoB[2]:.3f} {'✓' if abs(pb.rhoB[2] - 0.02) < 0.01 else '✗ WRONG!'}")
print(f"  rhoM[2]: Table=0.02, Actual={pb.rhoM[2]:.3f} {'✓' if abs(pb.rhoM[2] - 0.02) < 0.01 else '✗ WRONG!'}")
print(f"  rhoP[2]: Table=0.01, Actual={pb.rhoP[2]:.3f} {'✓' if abs(pb.rhoP[2] - 0.01) < 0.01 else '✗ WRONG!'}")
print(f"  nu[1][2]: Table=0.005, Actual={pb.nu[1].get(2, 0.0):.4f} {'✓' if abs(pb.nu[1].get(2, 0.0) - 0.005) < 0.0001 else '✗ WRONG!'}")
print(f"  t_max: Table=200, Actual={pb.t_max:.1f} (but run_scenarios.py overrides to 216.0)")

# Scenario C
print("\nSCENARIO C:")
print(f"  beta0[1]: Table=0.28, Actual={pc.beta0[1]:.3f} {'✓' if abs(pc.beta0[1] - 0.28) < 0.01 else '✗ WRONG!'}")
print(f"  beta0[2]: Table=0.26, Actual={pc.beta0[2]:.3f} {'✓' if abs(pc.beta0[2] - 0.26) < 0.01 else '✗ WRONG!'}")
print(f"  rhoB[2]: Table=0.006, Actual={pc.rhoB[2]:.4f} {'✓' if abs(pc.rhoB[2] - 0.006) < 0.0001 else '✗ WRONG!'}")
print(f"  rhoM[2]: Table=0.006, Actual={pc.rhoM[2]:.3f} {'✓' if abs(pc.rhoM[2] - 0.006) < 0.01 else '✗ WRONG!'}")
print(f"  rhoP[2]: Table=0.001, Actual={pc.rhoP[2]:.3f} {'✓' if abs(pc.rhoP[2] - 0.001) < 0.01 else '✗ WRONG!'}")
print(f"  nu[1][2]: Table=0.0002, Actual={pc.nu[1].get(2, 0.0):.4f} {'✓' if abs(pc.nu[1].get(2, 0.0) - 0.0002) < 0.0001 else '✗ WRONG!'}")
print(f"  nu[2][1]: Table=0.0002, Actual={pc.nu[2].get(1, 0.0):.4f} {'✓' if abs(pc.nu[2].get(1, 0.0) - 0.0002) < 0.0001 else '✗ WRONG!'}")

print("\n" + "=" * 70)
