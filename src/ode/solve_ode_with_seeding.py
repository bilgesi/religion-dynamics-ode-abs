# ode/solve_ode_with_seeding.py
"""
ODE solver with support for scheduled seeding events.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.model.indexing import make_index_map, pack_state, unpack_state
from src.model.params import ModelParams, validate_params
from src.ode.ode_system import dydt


def run_ode_with_seeding(
    params: ModelParams,
    S0: float,
    B0: Dict[int, float],
    M0: Dict[int, float],
    P0: Dict[int, float],
    seeding_events: List[Tuple[float, int, Dict[str, int]]],  # [(time, religion, {"B": count, "M": count, "P": count})]
    t_eval: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run ODE with scheduled seeding events.
    
    seeding_events: List of (time, religion_id, {"B": count, "M": count, "P": count})
    """
    validate_params(params)
    rels = sorted(params.religions)
    index_map = make_index_map(rels)

    if t_eval is None:
        t_eval = np.arange(0.0, params.t_max + params.dt, params.dt, dtype=float)
    
    # Sort seeding events by time
    seeding_events = sorted(seeding_events, key=lambda x: x[0])
    
    # Split time into segments at seeding events
    segments = []
    current_t = float(t_eval[0])
    current_y = pack_state(S0, B0, M0, P0, index_map)
    
    # Handle t=0 seeding by adding to initial state
    for seed_time, seed_r, seed_counts in seeding_events:
        if abs(seed_time - current_t) < 1e-6:  # t=0 seeding
            S, B, M, P = unpack_state(current_y, rels, index_map)
            B_add = float(seed_counts.get("B", 0))
            M_add = float(seed_counts.get("M", 0))
            P_add = float(seed_counts.get("P", 0))
            B[seed_r] += B_add
            M[seed_r] += M_add
            P[seed_r] += P_add
            # Remove from S to keep total population constant
            S -= (B_add + M_add + P_add)
            current_y = pack_state(S, B, M, P, index_map)
    
    for seed_time, seed_r, seed_counts in seeding_events:
        if seed_time <= current_t + 1e-6 or seed_time > params.t_max:
            continue
        
        # Segment before seeding
        t_seg = t_eval[(t_eval >= current_t) & (t_eval < seed_time)]
        if len(t_seg) > 0:
            segments.append((current_t, seed_time, current_y, t_seg))
        
        # Apply seeding: add agents to current state
        S, B, M, P = unpack_state(current_y, rels, index_map)
        B_add = float(seed_counts.get("B", 0))
        M_add = float(seed_counts.get("M", 0))
        P_add = float(seed_counts.get("P", 0))
        B[seed_r] += B_add
        M[seed_r] += M_add
        P[seed_r] += P_add
        # Remove from S to keep total population constant
        S -= (B_add + M_add + P_add)
        current_y = pack_state(S, B, M, P, index_map)
        current_t = seed_time
    
    # Final segment
    t_seg = t_eval[t_eval >= current_t]
    if len(t_seg) > 0:
        segments.append((current_t, params.t_max, current_y, t_seg))
    
    # Solve each segment
    all_t = []
    all_Y = []
    
    for t0_seg, t1_seg, y0_seg, t_eval_seg in segments:
        def f(t, y):
            return dydt(float(t), y, params, index_map)
        
        # Try SciPy; fallback to RK4 if missing
        try:
            from scipy.integrate import solve_ivp  # type: ignore
            sol = solve_ivp(
                fun=lambda t, y: f(t, y),
                t_span=(float(t0_seg), float(t1_seg)),
                y0=y0_seg,
                t_eval=t_eval_seg,
                vectorized=False,
                rtol=1e-6,
                atol=1e-9,
            )
            T_seg = sol.t
            Y_seg = sol.y.T
        except Exception:
            # Fallback RK4
            T_seg = t_eval_seg
            Y_seg = []
            y = y0_seg.astype(float).copy()
            dt = float(params.dt)
            t = float(t0_seg)
            for target_t in t_eval_seg:
                while t < target_t - 1e-12:
                    h = min(dt, target_t - t)
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h*k1/2)
                    k3 = f(t + h/2, y + h*k2/2)
                    k4 = f(t + h, y + h*k3)
                    y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
                    t += h
                Y_seg.append(y.copy())
            Y_seg = np.stack(Y_seg, axis=0)
        
        all_t.append(T_seg)
        all_Y.append(Y_seg)
    
    # Concatenate results
    T = np.concatenate(all_t)
    Y = np.concatenate(all_Y, axis=0)
    
    # Remove duplicates (at seeding times)
    unique_idx = np.unique(T, return_index=True)[1]
    T = T[unique_idx]
    Y = Y[unique_idx]
    
    # Unpack series
    out = {"t": T.tolist(), "S": Y[:, index_map["S"]].tolist(), "B": {}, "M": {}, "P": {}}
    for r in rels:
        out["B"][str(r)] = Y[:, index_map[f"B_{r}"]].tolist()
        out["M"][str(r)] = Y[:, index_map[f"M_{r}"]].tolist()
        out["P"][str(r)] = Y[:, index_map[f"P_{r}"]].tolist()
    return out
