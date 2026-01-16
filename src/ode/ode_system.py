"""
ODE system definition for religious dynamics.

Implements the right-hand side of the differential equations governing
population compartment flows. Core module.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.model.indexing import unpack_state
from src.model.params import ModelParams
from src.model.rates import all_lambdas, total_population
from src.model.context import rhoB_eff, rhoM_eff, rhoP_eff, q_eff


def dydt(t: float, y: np.ndarray, params: ModelParams, index_map: Dict[str, int]) -> np.ndarray:
    """
    Implements the Overleaf extended multi-strain model (S, B_r, M_r, P_r).
    Matches the structure we extracted from your screenshots:

    - lambda_r(t) = beta_r(t) * M_r / N
    - X_r = S + sum_{j!=r}(B_j + M_j)
    - inflow split into B_r vs M_r via q_r
    - cross-conversion outflow from B_r and M_r via sum_{l!=r} lambda_l
    - B<->M transitions (sigma, kappa)
    - B/M -> P transitions (tauB, tauM)
    - disaffiliation B/M/P -> S via rhoB/rhoM/rhoP
    - births/deaths b, mu applied to each compartment
    - mutation flows on missionaries via nu matrix

    NOTE: This is a faithful implementation of the ODE mechanism; we do not add any extra influence of P.
    """
    rels = params.religions
    S, B, M, P = unpack_state(y, rels, index_map)

    lambdas = all_lambdas(t, S, B, M, P, params)
    sum_lambda_all = sum(lambdas.values())

    # Precompute sums of B+M over all religions
    BM_total = sum(B[r] + M[r] for r in rels)

    # dS/dt
    # bS - mu S - S * sum_k lambda_k + sum_k (rhoB_k B_k + rhoM_k M_k + rhoP_k P_k)
    dS = params.b * S - params.mu * S - S * sum_lambda_all
    dS += sum(rhoB_eff(t, r, params) * B[r] + rhoM_eff(t, r, params) * M[r] + rhoP_eff(t, r, params) * P[r] for r in rels)

    dB: Dict[int, float] = {}
    dM: Dict[int, float] = {}
    dP: Dict[int, float] = {}

    # Mutation terms for missionaries
    # mut_in[r] = sum_{l!=r} nu[l][r] * M_l
    # mut_out[r] = sum_{l!=r} nu[r][l] * M_r
    mut_in = {r: 0.0 for r in rels}
    mut_out = {r: 0.0 for r in rels}
    if params.nu is not None:
        for l in rels:
            if l not in params.nu:
                continue
            for r, val in params.nu[l].items():
                if r == l:
                    continue
                if r in mut_in:
                    mut_in[r] += val * M[l]
        for r in rels:
            if r in params.nu:
                out_rate = sum(val for l, val in params.nu[r].items() if l != r)
                mut_out[r] = out_rate * M[r]

    for r in rels:
        # X_r = S + sum_{j!=r}(B_j + M_j)
        X_r = S + (BM_total - (B[r] + M[r]))

        # cross hazard faced by r-members from all other religions
        sum_lambda_others = sum(lambdas[l] for l in rels if l != r)

        # dB_r/dt
        q_r = q_eff(t, r, params)
        rhoB_r = rhoB_eff(t, r, params)
        dBr = params.b * B[r] - params.mu * B[r]
        dBr += (1.0 - q_r) * lambdas[r] * X_r
        dBr -= B[r] * sum_lambda_others
        dBr -= params.sigma[r] * B[r]
        dBr += params.kappa[r] * M[r]
        dBr -= rhoB_r * B[r]
        dBr -= params.tauB[r] * B[r]
        dB[r] = dBr

        # dM_r/dt
        rhoM_r = rhoM_eff(t, r, params)
        dMr = params.b * M[r] - params.mu * M[r]
        dMr += q_r * lambdas[r] * X_r
        dMr -= M[r] * sum_lambda_others
        dMr += params.sigma[r] * B[r]
        dMr -= params.kappa[r] * M[r]
        dMr -= rhoM_r * M[r]
        dMr -= params.tauM[r] * M[r]
        # mutation
        dMr += mut_in[r]
        dMr -= mut_out[r]
        dM[r] = dMr

        # dP_r/dt
        rhoP_r = rhoP_eff(t, r, params)
        dPr = params.b * P[r] - params.mu * P[r]
        dPr += params.tauB[r] * B[r]
        dPr += params.tauM[r] * M[r]
        dPr -= rhoP_r * P[r]
        dP[r] = dPr

    # Pack derivative
    dy = np.zeros_like(y, dtype=float)
    dy[index_map["S"]] = dS
    for r in rels:
        dy[index_map[f"B_{r}"]] = dB[r]
        dy[index_map[f"M_{r}"]] = dM[r]
        dy[index_map[f"P_{r}"]] = dP[r]
    return dy
