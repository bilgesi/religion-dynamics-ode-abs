"""
Context-dependent parameter modifiers.

Computes time-varying and state-dependent effective parameters (beta, rho, q)
including piecewise schedules and exogenous effects. Core module.
"""
from __future__ import annotations

from typing import Dict

from .params import ModelParams


def Z_state(t: float, r: int, params: ModelParams) -> float:
    """
    Placeholder exogenous index. Keep simple unless Overleaf defines a specific schedule.
    For now, return 0 (no state effect).
    """
    return 0.0


def Z_media(t: float, r: int, params: ModelParams) -> float:
    """
    Placeholder exogenous index. Keep simple unless Overleaf defines a specific schedule.
    For now, return 0 (no media effect).
    """
    return 0.0


def beta_r(t: float, r: int, params: ModelParams) -> float:
    """
    Effective beta for religion r at time t.

    If piecewise enabled for r -> uses beta0_pre/post based on t_crash[r].
    If context is disabled -> beta0[r].
    If enabled -> beta0[r] * max(0, 1 + a_state*Z_state + a_media*Z_media).
    """
    # Check for piecewise parameters first
    if params.t_crash is not None and r in params.t_crash:
        t_crash_r = params.t_crash[r]
        if params.beta0_pre is not None and r in params.beta0_pre:
            if t < t_crash_r:
                base = params.beta0_pre[r]
            else:
                # Use post if available, otherwise use base beta0
                if params.beta0_post is not None and r in params.beta0_post:
                    base = params.beta0_post[r]
                else:
                    base = params.beta0[r]
        else:
            base = params.beta0[r]
    else:
        base = params.beta0[r]
    
    if not params.context_enabled:
        return base

    z_s = Z_state(t, r, params)
    z_m = Z_media(t, r, params)
    factor = 1.0 + params.a_state * z_s + params.a_media * z_m
    if factor < 0.0:
        factor = 0.0
    return base * factor


def rhoB_eff(t: float, r: int, params: ModelParams) -> float:
    """Effective rhoB for religion r at time t (piecewise if enabled)."""
    if params.t_crash is not None and r in params.t_crash:
        t_crash_r = params.t_crash[r]
        if params.rhoB_pre is not None and r in params.rhoB_pre:
            if t < t_crash_r:
                return params.rhoB_pre[r]
            else:
                # Use post if available, otherwise use base rhoB
                if params.rhoB_post is not None and r in params.rhoB_post:
                    return params.rhoB_post[r]
                else:
                    return params.rhoB[r]
    return params.rhoB[r]


def rhoM_eff(t: float, r: int, params: ModelParams) -> float:
    """Effective rhoM for religion r at time t (piecewise if enabled)."""
    if params.t_crash is not None and r in params.t_crash:
        t_crash_r = params.t_crash[r]
        if params.rhoM_pre is not None and r in params.rhoM_pre:
            if t < t_crash_r:
                return params.rhoM_pre[r]
            else:
                # Use post if available, otherwise use base rhoM
                if params.rhoM_post is not None and r in params.rhoM_post:
                    return params.rhoM_post[r]
                else:
                    return params.rhoM[r]
    return params.rhoM[r]


def rhoP_eff(t: float, r: int, params: ModelParams) -> float:
    """Effective rhoP for religion r at time t (piecewise if enabled)."""
    if params.t_crash is not None and r in params.t_crash:
        t_crash_r = params.t_crash[r]
        if params.rhoP_pre is not None and r in params.rhoP_pre:
            if t < t_crash_r:
                return params.rhoP_pre[r]
            else:
                # Use post if available, otherwise use base rhoP
                if params.rhoP_post is not None and r in params.rhoP_post:
                    return params.rhoP_post[r]
                else:
                    return params.rhoP[r]
    return params.rhoP[r]


def q_eff(t: float, r: int, params: ModelParams) -> float:
    """Effective q for religion r at time t (piecewise if enabled)."""
    if params.t_crash is not None and r in params.t_crash:
        t_crash_r = params.t_crash[r]
        if params.q_pre is not None and r in params.q_pre:
            if t < t_crash_r:
                return params.q_pre[r]
            else:
                return params.q[r]  # q doesn't have post, use base
    return params.q[r]
