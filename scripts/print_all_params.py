#!/usr/bin/env python3
"""Print all parameters for manual table verification."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import scenario_A, scenario_B, scenario_C

pa, ia = scenario_A()
pb, ib = scenario_B()
pc, ic = scenario_C()

print("=" * 80)
print("SCENARIO A - ALL VALUES")
print("=" * 80)
print(f"Demography: b={pa.b}, mu={pa.mu}")
print(f"Religion 1: beta0={pa.beta0[1]}, q={pa.q[1]}, sigma={pa.sigma[1]}, kappa={pa.kappa[1]}")
print(f"  tauB={pa.tauB[1]}, tauM={pa.tauM[1]}, rhoB={pa.rhoB[1]}, rhoM={pa.rhoM[1]}, rhoP={pa.rhoP[1]}")
print(f"Religion 2: beta0={pa.beta0[2]}, q={pa.q[2]}, sigma={pa.sigma[2]}, kappa={pa.kappa[2]}")
print(f"  tauB={pa.tauB[2]}, tauM={pa.tauM[2]}, rhoB={pa.rhoB[2]}, rhoM={pa.rhoM[2]}, rhoP={pa.rhoP[2]}")
print(f"Mutation: nu[1][2]={pa.nu[1].get(2, 0.0)}, nu[2][1]={pa.nu[2].get(1, 0.0)}")
print(f"Initial: S0={ia['S0']}, B0={ia['B0']}, M0={ia['M0']}, P0={ia['P0']}")
print(f"Simulation: dt={pa.dt}, t_max={pa.t_max}")

print("\n" + "=" * 80)
print("SCENARIO B - ALL VALUES")
print("=" * 80)
print(f"Demography: b={pb.b}, mu={pb.mu}")
print(f"Religion 1: beta0={pb.beta0[1]}, q={pb.q[1]}, sigma={pb.sigma[1]}, kappa={pb.kappa[1]}")
print(f"  tauB={pb.tauB[1]}, tauM={pb.tauM[1]}, rhoB={pb.rhoB[1]}, rhoM={pb.rhoM[1]}, rhoP={pb.rhoP[1]}")
print(f"Religion 2: beta0={pb.beta0[2]}, q={pb.q[2]}, sigma={pb.sigma[2]}, kappa={pb.kappa[2]}")
print(f"  tauB={pb.tauB[2]}, tauM={pb.tauM[2]}, rhoB={pb.rhoB[2]}, rhoM={pb.rhoM[2]}, rhoP={pb.rhoP[2]}")
print(f"Religion 3: beta0={pb.beta0[3]}, q={pb.q[3]}, sigma={pb.sigma[3]}, kappa={pb.kappa[3]}")
print(f"  tauB={pb.tauB[3]}, tauM={pb.tauM[3]}, rhoB={pb.rhoB[3]}, rhoM={pb.rhoM[3]}, rhoP={pb.rhoP[3]}")
print(f"Mutation: nu[1][2]={pb.nu[1].get(2, 0.0)} (all should be 0)")
print(f"Initial: S0={ib['S0']}, B0={ib['B0']}, M0={ib['M0']}, P0={ib['P0']}")
print(f"Simulation: dt={pb.dt}, t_max={pb.t_max}")

print("\n" + "=" * 80)
print("SCENARIO C - ALL VALUES")
print("=" * 80)
print(f"Demography: b={pc.b}, mu={pc.mu}")
print(f"Religion 1: beta0={pc.beta0[1]}, q={pc.q[1]}, sigma={pc.sigma[1]}, kappa={pc.kappa[1]}")
print(f"  tauB={pc.tauB[1]}, tauM={pc.tauM[1]}, rhoB={pc.rhoB[1]}, rhoM={pc.rhoM[1]}, rhoP={pc.rhoP[1]}")
print(f"Religion 2: beta0={pc.beta0[2]}, q={pc.q[2]}, sigma={pc.sigma[2]}, kappa={pc.kappa[2]}")
print(f"  tauB={pc.tauB[2]}, tauM={pc.tauM[2]}, rhoB={pc.rhoB[2]}, rhoM={pc.rhoM[2]}, rhoP={pc.rhoP[2]}")
print(f"Mutation: nu[1][2]={pc.nu[1].get(2, 0.0)}, nu[2][1]={pc.nu[2].get(1, 0.0)}")
print(f"Initial: S0={ic['S0']}, B0={ic['B0']}, M0={ic['M0']}, P0={ic['P0']}")
print(f"Simulation: dt={pc.dt}, t_max={pc.t_max}")

print("\n" + "=" * 80)
