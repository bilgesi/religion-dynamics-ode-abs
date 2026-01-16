"""
ODE solver for religious dynamics model.

Integrates the mean-field equations using scipy.solve_ivp with RK4 fallback.
Core module.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.model.indexing import make_index_map, pack_state, unpack_state
from src.model.params import ModelParams, validate_params
from .ode_system import dydt


def _solve_ivp_fallback(fun, t_span, y0, t_eval):
    """
    Simple RK4 fallback if SciPy is unavailable.
    """
    t0, t1 = t_span
    y = y0.astype(float).copy()
    ys = [y.copy()]
    ts = [t_eval[0]]
    dt = float(t_eval[1] - t_eval[0]) if len(t_eval) > 1 else (t1 - t0)
    t = t0
    for k in range(1, len(t_eval)):
        target_t = float(t_eval[k])
        while t < target_t - 1e-12:
            h = min(dt, target_t - t)
            k1 = fun(t, y)
            k2 = fun(t + h/2, y + h*k1/2)
            k3 = fun(t + h/2, y + h*k2/2)
            k4 = fun(t + h, y + h*k3)
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        ys.append(y.copy())
        ts.append(target_t)
    return np.array(ts), np.stack(ys, axis=0)


def run_ode(
    params: ModelParams,
    S0: float,
    B0: Dict[int, float],
    M0: Dict[int, float],
    P0: Dict[int, float],
    t_eval: Optional[np.ndarray] = None,
) -> Dict:
    validate_params(params)
    rels = sorted(params.religions)
    index_map = make_index_map(rels)

    if t_eval is None:
        t_eval = np.arange(0.0, params.t_max + params.dt, params.dt, dtype=float)

    y0 = pack_state(S0, B0, M0, P0, index_map)

    def f(t, y):
        return dydt(float(t), y, params, index_map)

    # Try SciPy; fallback to RK4 if missing
    try:
        from scipy.integrate import solve_ivp  # type: ignore
        sol = solve_ivp(
            fun=lambda t, y: f(t, y),
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            t_eval=t_eval,
            vectorized=False,
            rtol=1e-6,
            atol=1e-9,
        )
        T = sol.t
        Y = sol.y.T
    except Exception:
        T, Y = _solve_ivp_fallback(lambda t, y: f(t, y), (float(t_eval[0]), float(t_eval[-1])), y0, t_eval)

    # Unpack series
    out = {"t": T.tolist(), "S": Y[:, index_map["S"]].tolist(), "B": {}, "M": {}, "P": {}}
    for r in rels:
        out["B"][str(r)] = Y[:, index_map[f"B_{r}"]].tolist()
        out["M"][str(r)] = Y[:, index_map[f"M_{r}"]].tolist()
        out["P"][str(r)] = Y[:, index_map[f"P_{r}"]].tolist()
    return out
