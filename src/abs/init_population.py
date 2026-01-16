"""
Population initialization for ABS simulations.

Creates initial agent arrays from compartment counts (S, B, M, P per religion).
Core module.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .agent import ROLE_S, ROLE_B, ROLE_M, ROLE_P


def init_population_from_counts(
    S0: int,
    B0: Dict[int, int],
    M0: Dict[int, int],
    P0: Dict[int, int],
    religions: List[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      roles: int array of shape (N,)
      rel_ids: int array of shape (N,) (0 for S, else religion id)
    """
    rels = sorted(religions)
    N = int(S0 + sum(B0.get(r, 0) for r in rels) + sum(M0.get(r, 0) for r in rels) + sum(P0.get(r, 0) for r in rels))
    roles = np.empty(N, dtype=np.int8)
    rel_ids = np.empty(N, dtype=np.int32)

    idx = 0
    # S
    roles[idx:idx + S0] = ROLE_S
    rel_ids[idx:idx + S0] = 0
    idx += S0

    # B/M/P per religion
    for r in rels:
        nb = int(B0.get(r, 0))
        nm = int(M0.get(r, 0))
        np_ = int(P0.get(r, 0))

        if nb:
            roles[idx:idx + nb] = ROLE_B
            rel_ids[idx:idx + nb] = r
            idx += nb
        if nm:
            roles[idx:idx + nm] = ROLE_M
            rel_ids[idx:idx + nm] = r
            idx += nm
        if np_:
            roles[idx:idx + np_] = ROLE_P
            rel_ids[idx:idx + np_] = r
            idx += np_

    # Shuffle population (well-mixed, no spatial structure)
    perm = rng.permutation(N)
    return roles[perm], rel_ids[perm]
