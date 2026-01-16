"""
Model parameter container and validation.

Defines the ModelParams dataclass holding all demographic, conversion, and
transition rate parameters for both ODE and ABS models. Core module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ModelParams:
    """
    Single-model parameter container (ODE is the formal model; ABS implements it stochastically).

    Religions are indexed by integers: 1..m
    Role compartments are: S, B_r, M_r, P_r (multi-strain).
    """
    religions: List[int]

    # Demography
    b: float
    mu: float

    # Per-religion parameters (dict keyed by religion id)
    beta0: Dict[int, float]   # baseline beta (beta_r(t) may be beta0 * f(Z))
    q: Dict[int, float]       # fraction entering M upon conversion into r
    sigma: Dict[int, float]   # B_r -> M_r
    kappa: Dict[int, float]   # M_r -> B_r
    tauB: Dict[int, float]    # B_r -> P_r
    tauM: Dict[int, float]    # M_r -> P_r
    rhoB: Dict[int, float]    # B_r -> S
    rhoM: Dict[int, float]    # M_r -> S
    rhoP: Dict[int, float]    # P_r -> S (disaffiliation; ODE screenshot includes this term)

    # Mutation matrix nu[r][l] (r!=l): M_r -> M_l
    nu: Optional[Dict[int, Dict[int, float]]] = None

    # Context modulation controls (optional)
    context_enabled: bool = False
    # If enabled, beta_r(t) = beta0_r * max(0, 1 + a_state*Z_state(t,r) + a_media*Z_media(t,r))
    a_state: float = 0.0
    a_media: float = 0.0

    # Piecewise time-varying parameters (for Scenario B cults)
    # t_crash[r] = crash time for religion r (None if not piecewise)
    t_crash: Optional[Dict[int, float]] = None
    # Pre-crash parameters (used when t < t_crash[r])
    beta0_pre: Optional[Dict[int, float]] = None
    q_pre: Optional[Dict[int, float]] = None
    rhoB_pre: Optional[Dict[int, float]] = None
    rhoM_pre: Optional[Dict[int, float]] = None
    rhoP_pre: Optional[Dict[int, float]] = None
    # Post-crash parameters (used when t >= t_crash[r])
    beta0_post: Optional[Dict[int, float]] = None
    rhoB_post: Optional[Dict[int, float]] = None
    rhoM_post: Optional[Dict[int, float]] = None
    rhoP_post: Optional[Dict[int, float]] = None

    # Simulation controls (ABS/ODE time grid)
    dt: float = 0.1
    t_max: float = 200.0


def validate_params(p: ModelParams) -> None:
    rels = set(p.religions)
    if len(rels) != len(p.religions):
        raise ValueError("religions list contains duplicates")

    required_dicts = [
        ("beta0", p.beta0),
        ("q", p.q),
        ("sigma", p.sigma),
        ("kappa", p.kappa),
        ("tauB", p.tauB),
        ("tauM", p.tauM),
        ("rhoB", p.rhoB),
        ("rhoM", p.rhoM),
        ("rhoP", p.rhoP),
    ]
    for name, d in required_dicts:
        missing = [r for r in p.religions if r not in d]
        if missing:
            raise ValueError(f"Missing {name} for religions: {missing}")

    # bounds
    for r in p.religions:
        if not (0.0 <= p.q[r] <= 1.0):
            raise ValueError(f"q[{r}] must be in [0,1]")
        if p.beta0[r] < 0:
            raise ValueError(f"beta0[{r}] must be >= 0")

    if p.dt <= 0 or p.t_max <= 0:
        raise ValueError("dt and t_max must be positive")

    if p.nu is not None:
        for r in p.religions:
            if r not in p.nu:
                continue
            for l, val in p.nu[r].items():
                if l == r:
                    raise ValueError("nu[r][r] must not be set")
                if val < 0:
                    raise ValueError("nu entries must be >= 0")


def copy_with_dt(p: ModelParams, dt: float, t_max: float) -> ModelParams:
    # convenience helper
    return ModelParams(
        religions=list(p.religions),
        b=p.b,
        mu=p.mu,
        beta0=dict(p.beta0),
        q=dict(p.q),
        sigma=dict(p.sigma),
        kappa=dict(p.kappa),
        tauB=dict(p.tauB),
        tauM=dict(p.tauM),
        rhoB=dict(p.rhoB),
        rhoM=dict(p.rhoM),
        rhoP=dict(p.rhoP),
        nu=None if p.nu is None else {rr: dict(p.nu[rr]) for rr in p.nu},
        context_enabled=p.context_enabled,
        a_state=p.a_state,
        a_media=p.a_media,
        dt=dt,
        t_max=t_max,
    )
