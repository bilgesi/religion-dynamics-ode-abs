"""
State vector indexing utilities for the ODE system.

Maps compartment names (S, B_r, M_r, P_r) to array indices for vectorized
ODE integration. Core module.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def make_index_map(religions: List[int]) -> Dict[str, int]:
    """
    State vector y layout:
      y[0] = S
      then for each r in religions (sorted):
        B_r, M_r, P_r
    """
    rels = sorted(religions)
    idx: Dict[str, int] = {"S": 0}
    k = 1
    for r in rels:
        idx[f"B_{r}"] = k; k += 1
        idx[f"M_{r}"] = k; k += 1
        idx[f"P_{r}"] = k; k += 1
    return idx


def pack_state(S: float, B: Dict[int, float], M: Dict[int, float], P: Dict[int, float], index_map: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(index_map), dtype=float)
    y[index_map["S"]] = S
    for key, i in index_map.items():
        if key == "S":
            continue
        kind, r_str = key.split("_")
        r = int(r_str)
        if kind == "B":
            y[i] = float(B.get(r, 0.0))
        elif kind == "M":
            y[i] = float(M.get(r, 0.0))
        elif kind == "P":
            y[i] = float(P.get(r, 0.0))
    return y


def unpack_state(y: np.ndarray, religions: List[int], index_map: Dict[str, int]) -> Tuple[float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    S = float(y[index_map["S"]])
    B = {r: float(y[index_map[f"B_{r}"]]) for r in religions}
    M = {r: float(y[index_map[f"M_{r}"]]) for r in religions}
    P = {r: float(y[index_map[f"P_{r}"]]) for r in religions}
    return S, B, M, P
