"""
Predefined scenario configurations for stylized experiments.

Defines parameter sets and initial conditions for Scenarios A, B, and C,
representing different religious dynamics regimes. Core module.
"""
from __future__ import annotations

from typing import Dict, Tuple

from src.model.params import ModelParams


def scenario_A() -> Tuple[ModelParams, Dict]:
    """
    A) Large religion from other large religion
    - Two religions: parent=1, new=2 (predefined)
    - Mutation from 1 -> 2 on missionaries enables emergence.
    """
    religions = [1, 2]
    params = ModelParams(
        religions=religions,
        b=0.01,
        mu=0.01,

        beta0={1: 0.03, 2: 0.45},  # r1 beta reduced ~10x to ensure some people remain without religion
        q={1: 0.05, 2: 0.10},
        sigma={1: 0.02, 2: 0.03},
        kappa={1: 0.01, 2: 0.01},
        tauB={1: 0.002, 2: 0.003},
        tauM={1: 0.003, 2: 0.004},
        rhoB={1: 0.005, 2: 0.004},
        rhoM={1: 0.004, 2: 0.003},
        rhoP={1: 0.001, 2: 0.001},

        nu={
            1: {2: 0.002},  # schism from parent to new
            2: {1: 0.0},
        },

        context_enabled=False,
        dt=0.1,
        t_max=200.0,
    )

    init = {
        "S0": 6000,
        "B0": {1: 3000, 2: 0},
        "M0": {1: 200,  2: 0},
        "P0": {1: 800,  2: 0},
    }
    return params, init


def scenario_B() -> Tuple[ModelParams, Dict]:
    """
    B) Small cults come and go with piecewise time-varying parameters
    - One main religion + multiple potential cult strains (2..6)
    - Strains are seeded at scheduled times: r2=0, r3=16, r4=48, r5=112, r6=167
    - Each cult uses piecewise parameters: pre-crash (high growth) then post-crash (rapid decay)
    - Crash times: t_crash = {2:60, 3:76, 4:108, 5:172, 6:207} (weeks)
    - Mutation disabled (nu=0) - seeding used instead.
    """
    religions = [1, 2, 3, 4, 5, 6]
    # r1 parameters (dominant strain) - constant, reduced beta to allow cults to emerge
    beta0 = {1: 0.04}  # Reduced by ~8x (0.33 -> 0.04) to allow cults to emerge while r=1 remains dominant
    q = {1: 0.05}
    sigma = {1: 0.02}
    kappa = {1: 0.02}
    tauB = {1: 0.002}
    tauM = {1: 0.003}
    rhoB = {1: 0.006}
    rhoM = {1: 0.006}
    rhoP = {1: 0.001}

    # Base parameters for cults (r=2..6) - used as fallback/post-crash
    for r in religions[1:]:
        beta0[r] = 0.02  # Post-crash: low beta
        q[r] = 0.02      # Low q to prevent takeover (reduced from 0.03)
        sigma[r] = 0.005  # Low B->M production (reduced from 0.01)
        kappa[r] = 0.02  # M->B transition rate
        tauB[r] = 0.0003
        tauM[r] = 0.0003
        rhoB[r] = 0.05   # Post-crash: moderate disaffiliation
        rhoM[r] = 0.15   # Post-crash: moderate M disaffiliation
        rhoP[r] = 0.4    # Post-crash: high P disaffiliation

    # Piecewise parameters for cults (r=2..6)
    # Crash times: optimized to allow proper die-out (r=6 needs ~26 weeks after crash at t_max=216)
    t_crash = {2: 40.0, 3: 60.0, 4: 90.0, 5: 145.0, 6: 190.0}
    beta0_pre = {}
    q_pre = {}
    rhoB_pre = {}
    rhoM_pre = {}
    rhoP_pre = {}
    
    for r in religions[1:]:
        beta0_pre[r] = 0.75  # Pre-crash: higher beta to reach ~0.2 peak (increased from 0.4)
        q_pre[r] = 0.02      # Pre-crash: low q to prevent takeover
        rhoB_pre[r] = 0.01   # Pre-crash: low disaffiliation
        rhoM_pre[r] = 0.01   # Pre-crash: low disaffiliation
        rhoP_pre[r] = 0.01   # Pre-crash: low disaffiliation

    # Disable mutation (seeding used instead)
    nu = {r: {} for r in religions}
    for r in religions:
        for s in religions:
            if r != s:
                nu[r][s] = 0.0

    params = ModelParams(
        religions=religions,
        b=0.01,
        mu=0.01,
        beta0=beta0,
        q=q,
        sigma=sigma,
        kappa=kappa,
        tauB=tauB,
        tauM=tauM,
        rhoB=rhoB,
        rhoM=rhoM,
        rhoP=rhoP,
        nu=nu,
        context_enabled=False,
        t_crash=t_crash,
        beta0_pre=beta0_pre,
        q_pre=q_pre,
        rhoB_pre=rhoB_pre,
        rhoM_pre=rhoM_pre,
        rhoP_pre=rhoP_pre,
        beta0_post=None,  # Use base beta0 as post
        rhoB_post=None,   # Use base rhoB as post
        rhoM_post=None,   # Use base rhoM as post
        rhoP_post=None,   # Use base rhoP as post
        dt=0.1,
        t_max=216.0,
    )

    init = {
        "S0": 5000,  # Reduced from 6400 to 5000 to accommodate r=1 increase to 5000
        "B0": {1: 4200, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},  # Increased to make total r=1 = 5000
        "M0": {1: 300,  2: 0, 3: 0, 4: 0, 5: 0, 6: 0},  # Increased to make total r=1 = 5000
        "P0": {1: 500,  2: 0, 3: 0, 4: 0, 5: 0, 6: 0},  # Increased to make total r=1 = 5000
    }
    return params, init


def scenario_C() -> Tuple[ModelParams, Dict]:
    """
    C) Second religion rises from 0, reaches small value, then dies out
    - Two religions: r=1 starts with population, r=2 starts with small seed (B2=100)
    - r=2 rises to ~0.036, then dies out due to high disaffiliation
    - Parameters tuned via sweep to achieve peak >= 0.02 and final <= 0.002
    """
    religions = [1, 2]
    params = ModelParams(
        religions=religions,
        b=0.01,
        mu=0.01,

        beta0={1: 0.28, 2: 0.55},  # r2 higher beta to allow growth (tuned via sweep)
        q={1: 0.05, 2: 0.05},
        sigma={1: 0.018, 2: 0.018},
        kappa={1: 0.018, 2: 0.018},
        tauB={1: 0.0015, 2: 0.0015},
        tauM={1: 0.0020, 2: 0.0020},
        # r2 has higher disaffiliation rates to ensure it dies out (tuned via sweep)
        rhoB={1: 0.006, 2: 0.0225},  # r2 higher disaffiliation
        rhoM={1: 0.006, 2: 0.03},    # r2 higher disaffiliation
        rhoP={1: 0.001, 2: 0.003},   # r2 higher disaffiliation

        nu={
            1: {2: 0.0},  # mutation disabled (seeding used instead)
            2: {1: 0.0},  # r2 doesn't mutate back
        },

        context_enabled=False,
        dt=0.1,
        t_max=400.0,  # longer time to show rise and fall
    )

    init = {
        "S0": 6500,
        "B0": {1: 2000, 2: 100},  # r2 starts with small seed (tuned via sweep)
        "M0": {1: 120,  2: 0},    # r2 starts at 0
        "P0": {1: 180,  2: 0},    # r2 starts at 0
    }
    return params, init


SCENARIOS = {"A": scenario_A, "B": scenario_B, "C": scenario_C}
