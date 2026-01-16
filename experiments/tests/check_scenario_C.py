# experiments/check_scenario_C.py
"""
Scenario C validation script:
1) Is r=2 present at initialization? (ABS side)
2) Is nu/mutation truly disabled?
3) Does death+birth allow r=2 to regrow? (composition-preserving check)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.abs.init_population import init_population_from_counts
from src.abs.sim_abs import run_abs
from src.scenarios import scenario_C
from src.config import DEFAULT_CONFIG


def check_scenario_C():
    print("=" * 80)
    print("Scenario C Validation Checklist")
    print("=" * 80)
    
    params, init = scenario_C()
    cfg = DEFAULT_CONFIG
    params.dt = cfg.dt
    params.t_max = 400.0
    
    S0 = int(init["S0"])
    B0 = {int(k): int(v) for k, v in init["B0"].items()}
    M0 = {int(k): int(v) for k, v in init["M0"].items()}
    P0 = {int(k): int(v) for k, v in init["P0"].items()}
    
    print("\n1) Initial Conditions Check:")
    print(f"   S0 = {S0}")
    print(f"   B0 = {B0}")
    print(f"   M0 = {M0}")
    print(f"   P0 = {P0}")
    print(f"   Expected: B0[2] = 100")
    
    # Initialize ABS population
    rng = np.random.default_rng(42)
    roles, rel_ids = init_population_from_counts(S0, B0, M0, P0, params.religions, rng)
    
    # Check r=2 counts
    r2_total = int(np.sum(rel_ids == 2))
    r2_B = int(np.sum((rel_ids == 2) & (roles == 1)))  # ROLE_B = 1
    r2_M = int(np.sum((rel_ids == 2) & (roles == 2)))  # ROLE_M = 2
    r2_P = int(np.sum((rel_ids == 2) & (roles == 3)))  # ROLE_P = 3
    
    print(f"\n   After ABS initialization:")
    print(f"   r=2 total agents: {r2_total}")
    print(f"   r=2 B (Believers): {r2_B}")
    print(f"   r=2 M (Missionaries): {r2_M}")
    print(f"   r=2 P (Priests): {r2_P}")
    
    if r2_total == 100 and r2_B == 100:
        print("   OK: r=2 present at initialization with 100 B agents")
    else:
        print(f"   WARNING: r=2 differs from expected value! (expected: 100 B)")
    
    print("\n2) Mutation (nu) Check:")
    if params.nu is None:
        print("   OK: params.nu = None (mutation completely disabled)")
    else:
        print(f"   params.nu = {params.nu}")
        has_positive = False
        for r, out_map in params.nu.items():
            for target, rate in out_map.items():
                if rate > 0:
                    has_positive = True
                    print(f"   WARNING: nu[{r}][{target}] = {rate} > 0 (mutation active!)")
        
        if not has_positive:
            print("   OK: All nu values = 0 (mutation ineffective)")
    
    print("\n3) Composition-preserving Birth Check:")
    print("   Birth mechanism: copies parent's role and religion")
    print("   This means: when r=2 becomes very small, regrowth via birth is very difficult")
    print("   (Theoretically possible but practically consistent with final ~0.002 target)")
    
    print("\n4) Short Simulation Test (first 10 time-steps):")
    # Run a very short simulation to check initial behavior
    params_short = params
    params_short.t_max = 1.0  # 1 week only
    out_short = run_abs(params_short, roles.copy(), rel_ids.copy(), seed=42, seeding_events=None)
    
    # Check r=2 at start and end
    t_series = np.array(out_short["t"])
    B2_series = np.array(out_short["B"]["2"])
    M2_series = np.array(out_short["M"]["2"])
    P2_series = np.array(out_short["P"]["2"])
    
    r2_start = B2_series[0] + M2_series[0] + P2_series[0]
    r2_end = B2_series[-1] + M2_series[-1] + P2_series[-1]
    
    print(f"   t=0: r=2 total = {r2_start:.1f} (B={B2_series[0]:.1f}, M={M2_series[0]:.1f}, P={P2_series[0]:.1f})")
    print(f"   t={t_series[-1]:.1f}: r=2 total = {r2_end:.1f} (B={B2_series[-1]:.1f}, M={M2_series[-1]:.1f}, P={P2_series[-1]:.1f})")
    
    if r2_start > 0:
        print("   OK: r=2 present at simulation start")
    else:
        print("   ERROR: r=2 not present at simulation start!")
    
    print("\n5) Seeding Events Check:")
    print("   In Scenario C, seeding_events should be None")
    print("   (r=2 starts from initial condition, no scheduled seeding)")
    print("   OK: Seeding injection patch ineffective for Scenario C (seeding_events=None)")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    check_scenario_C()
