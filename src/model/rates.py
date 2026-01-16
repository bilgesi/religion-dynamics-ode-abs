"""
Rate computation functions for the religious dynamics model.

Computes conversion rates (lambda), total population, and probability utilities.
Core module.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .context import beta_r
from .params import ModelParams


def total_population(S: float, B: Dict[int, float], M: Dict[int, float], P: Dict[int, float]) -> float:
    return float(S + sum(B.values()) + sum(M.values()) + sum(P.values()))


def lambda_r(t: float, r: int, S: float, B: Dict[int, float], M: Dict[int, float], P: Dict[int, float], params: ModelParams) -> float:
    """
    ODE screenshot definition:
      lambda_r(t) = beta_r(t) * M_r(t) / N(t)
    """
    N = total_population(S, B, M, P)
    if N <= 0.0:
        return 0.0
    return beta_r(t, r, params) * (float(M.get(r, 0.0)) / N)


def all_lambdas(t: float, S: float, B: Dict[int, float], M: Dict[int, float], P: Dict[int, float], params: ModelParams) -> Dict[int, float]:
    return {r: lambda_r(t, r, S, B, M, P, params) for r in params.religions}


def prob_from_rate(rate: float, dt: float) -> float:
    if rate <= 0.0:
        return 0.0
    # p = 1 - exp(-rate*dt)
    return float(1.0 - np.exp(-rate * dt))


def choose_weighted(keys, weights, rng: np.random.Generator) -> int:
    w = np.array(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        # fallback uniform
        return int(rng.choice(list(keys)))
    p = w / s
    return int(rng.choice(list(keys), p=p))


def conversion_hazard_S(lambdas: Dict[int, float]) -> float:
    return float(sum(lambdas.values()))


def conversion_hazard_for_religion(current_r: int, lambdas: Dict[int, float]) -> float:
    return float(sum(v for r, v in lambdas.items() if r != current_r))


def pick_target_religion_for_S(lambdas: Dict[int, float], rng: np.random.Generator) -> int:
    rels = list(lambdas.keys())
    w = [lambdas[r] for r in rels]
    return choose_weighted(rels, w, rng)


def pick_target_religion_for_r(current_r: int, lambdas: Dict[int, float], rng: np.random.Generator) -> int:
    rels = [r for r in lambdas.keys() if r != current_r]
    w = [lambdas[r] for r in rels]
    return choose_weighted(rels, w, rng)
