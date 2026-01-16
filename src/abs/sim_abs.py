"""
Agent-Based Simulation (ABS) engine.

Implements stochastic discrete-time simulation of religious dynamics with
birth, death, conversion, promotion, and apostasy events. Core module.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.model.params import ModelParams, validate_params
from src.model.rates import prob_from_rate, choose_weighted
from src.model.context import beta_r, rhoB_eff, rhoM_eff, rhoP_eff, q_eff
from .agent import ROLE_S, ROLE_B, ROLE_M, ROLE_P


def _count_compartments(roles: np.ndarray, rel_ids: np.ndarray, religions: List[int]) -> Dict:
    rels = sorted(religions)
    S = int(np.sum(roles == ROLE_S))
    B = {r: int(np.sum((roles == ROLE_B) & (rel_ids == r))) for r in rels}
    M = {r: int(np.sum((roles == ROLE_M) & (rel_ids == r))) for r in rels}
    P = {r: int(np.sum((roles == ROLE_P) & (rel_ids == r))) for r in rels}
    return {"S": S, "B": B, "M": M, "P": P}


def _compute_lambdas(t: float, counts: Dict, params: ModelParams) -> Dict[int, float]:
    rels = params.religions
    S = float(counts["S"])
    B = {r: float(counts["B"][r]) for r in rels}
    M = {r: float(counts["M"][r]) for r in rels}
    P = {r: float(counts["P"][r]) for r in rels}
    N = S + sum(B.values()) + sum(M.values()) + sum(P.values())
    if N <= 0.0:
        return {r: 0.0 for r in rels}
    return {r: beta_r(t, r, params) * (M[r] / N) for r in rels}


def _inject_agents(roles: np.ndarray, rel_ids: np.ndarray, r: int, add: dict, rng: np.random.Generator) -> tuple:
    """Convert S agents into B/M/P for scheduled seeding (ODE-consistent, N constant)."""
    nB = int(add.get("B", 0))
    nM = int(add.get("M", 0))
    nP = int(add.get("P", 0))
    total = nB + nM + nP
    if total <= 0:
        return roles, rel_ids

    s_idx = np.where(roles == ROLE_S)[0]
    if s_idx.size <= 0:
        return roles, rel_ids

    # If not enough S, scale B/M/P proportionally to preserve ratios (keeps N constant)
    if s_idx.size < total:
        n = s_idx.size
        # Preserve ratios: scale proportionally
        nB2 = int(round(n * (nB / total)))
        nM2 = int(round(n * (nM / total)))
        nP2 = n - nB2 - nM2  # Ensure nB2 + nM2 + nP2 = n
        nB, nM, nP = nB2, nM2, nP2
    else:
        n = total

    chosen = rng.choice(s_idx, size=n, replace=False)

    i = 0
    take = min(nB, n - i)
    if take > 0:
        idx = chosen[i:i+take]
        roles[idx] = ROLE_B
        rel_ids[idx] = int(r)
        i += take

    take = min(nM, n - i)
    if take > 0:
        idx = chosen[i:i+take]
        roles[idx] = ROLE_M
        rel_ids[idx] = int(r)
        i += take

    take = min(nP, n - i)
    if take > 0:
        idx = chosen[i:i+take]
        roles[idx] = ROLE_P
        rel_ids[idx] = int(r)
        i += take

    return roles, rel_ids


def run_abs(
    params: ModelParams,
    roles: np.ndarray,
    rel_ids: np.ndarray,
    seed: int,
    seeding_events: Optional[List[tuple]] = None,
) -> Dict:
    """
    Discrete-time hazard-based ABS implementing exactly the ODE mechanisms (well-mixed / mass-action).

    Order of updates per step (small dt assumed):
      1) conversion (S/B/M -> target religion) using lambdas
      2) B<->M transitions
      3) B/M -> P transitions
      4) disaffiliation (B/M/P -> S)
      5) death
      6) birth (composition-preserving)
      7) mutation on missionaries (nu)
    """
    validate_params(params)
    rng = np.random.default_rng(seed)
    rels = sorted(params.religions)

    dt = float(params.dt)
    T = int(np.floor(params.t_max / dt)) + 1
    t_series = np.linspace(0.0, dt * (T - 1), T)

    # Map seeding events to time steps (use >= logic to avoid float comparison issues)
    # Track which events have been applied to prevent duplicate applications
    applied_events = set()
    if seeding_events:
        # Sort events by time
        seeding_events = sorted(seeding_events, key=lambda x: x[0])

    # Time series storage
    S_ts = np.zeros(T, dtype=float)
    B_ts = {r: np.zeros(T, dtype=float) for r in rels}
    M_ts = {r: np.zeros(T, dtype=float) for r in rels}
    P_ts = {r: np.zeros(T, dtype=float) for r in rels}

    for ti, t in enumerate(t_series):
        # ---------- 0) Scheduled seeding (Scenario B) ----------
        # Apply seeding events when t >= seed_time (more robust than round())
        # MUST be before _count_compartments and _compute_lambdas
        if seeding_events:
            for event_idx, (t_event, r_seed, add) in enumerate(seeding_events):
                if event_idx in applied_events:
                    continue
                # Events are sorted by time, so if this one is too early, break
                if float(t) < float(t_event) - 1e-9:
                    break
                # Apply seeding event
                # Debug log for seeding events
                N_before = roles.shape[0]
                roles, rel_ids = _inject_agents(roles, rel_ids, r_seed, add, rng)
                N_after = roles.shape[0]
                nB = int(add.get("B", 0))
                nM = int(add.get("M", 0))
                nP = int(add.get("P", 0))
                # Sanity check: N must remain constant (conversion, not addition)
                if N_before != N_after:
                    print(f"  [WARNING] N changed during seeding: {N_before} -> {N_after}", flush=True)
                print(f"  [SEEDING] t={t:.3f}, r={r_seed}, B={nB}, M={nM}, P={nP}, N: {N_before} -> {N_after} (should be equal)", flush=True)
                applied_events.add(event_idx)

        counts = _count_compartments(roles, rel_ids, rels)
        S_ts[ti] = counts["S"]
        for r in rels:
            B_ts[r][ti] = counts["B"][r]
            M_ts[r][ti] = counts["M"][r]
            P_ts[r][ti] = counts["P"][r]

        lambdas = _compute_lambdas(float(t), counts, params)
        sum_lambda_all = sum(lambdas.values())

        # ---------- 1) Conversion (S/B/M only; P never switches religions) ----------
        # For speed: vector masks
        mask_S = (roles == ROLE_S)
        mask_B = (roles == ROLE_B)
        mask_M = (roles == ROLE_M)
        mask_P = (roles == ROLE_P)

        # S converts with hazard sum_lambda_all
        pS = prob_from_rate(sum_lambda_all, dt)
        if pS > 0:
            draws = rng.random(np.sum(mask_S))
            conv_S_idx = np.where(mask_S)[0][draws < pS]
            for idx in conv_S_idx:
                target = _pick_target_for_S(lambdas, rng)
                # new role split via q_eff[target]
                q_target = q_eff(float(t), target, params)
                if rng.random() < q_target:
                    roles[idx] = ROLE_M
                else:
                    roles[idx] = ROLE_B
                rel_ids[idx] = target

        # B and M cross-convert with hazard sum_{l!=r} lambda_l
        for role_mask, role_code in [(mask_B, ROLE_B), (mask_M, ROLE_M)]:
            idxs = np.where(role_mask)[0]
            if idxs.size == 0:
                continue
            # For each agent, compute hazard excluding current religion
            for idx in idxs:
                r0 = int(rel_ids[idx])
                if r0 == 0:
                    continue
                hazard = sum(lambdas[r] for r in rels if r != r0)
                p = prob_from_rate(hazard, dt)
                if p <= 0:
                    continue
                if rng.random() < p:
                    target = _pick_target_for_r(r0, lambdas, rng)
                    # upon entry into target, split via q_eff
                    q_target = q_eff(float(t), target, params)
                    roles[idx] = ROLE_M if (rng.random() < q_target) else ROLE_B
                    rel_ids[idx] = target

        # ---------- 2) B<->M transitions within same religion ----------
        idxB = np.where(roles == ROLE_B)[0]
        for idx in idxB:
            r = int(rel_ids[idx])
            p = prob_from_rate(params.sigma[r], dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_M

        idxM = np.where(roles == ROLE_M)[0]
        for idx in idxM:
            r = int(rel_ids[idx])
            p = prob_from_rate(params.kappa[r], dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_B

        # ---------- 3) Promotion to P ----------
        idxB = np.where(roles == ROLE_B)[0]
        for idx in idxB:
            r = int(rel_ids[idx])
            p = prob_from_rate(params.tauB[r], dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_P

        idxM = np.where(roles == ROLE_M)[0]
        for idx in idxM:
            r = int(rel_ids[idx])
            p = prob_from_rate(params.tauM[r], dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_P

        # ---------- 4) Disaffiliation to S (B/M/P -> S) ----------
        idxB = np.where(roles == ROLE_B)[0]
        for idx in idxB:
            r = int(rel_ids[idx])
            p = prob_from_rate(rhoB_eff(float(t), r, params), dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_S
                rel_ids[idx] = 0

        idxM = np.where(roles == ROLE_M)[0]
        for idx in idxM:
            r = int(rel_ids[idx])
            p = prob_from_rate(rhoM_eff(float(t), r, params), dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_S
                rel_ids[idx] = 0

        idxP = np.where(roles == ROLE_P)[0]
        for idx in idxP:
            r = int(rel_ids[idx])
            p = prob_from_rate(rhoP_eff(float(t), r, params), dt)
            if p > 0 and rng.random() < p:
                roles[idx] = ROLE_S
                rel_ids[idx] = 0

        # ---------- 5) Death ----------
        pD = prob_from_rate(params.mu, dt)
        if pD > 0:
            survive = rng.random(roles.shape[0]) >= pD
            roles = roles[survive]
            rel_ids = rel_ids[survive]

        # ---------- 6) Birth (composition-preserving) ----------
        # births ~ Poisson(b*N*dt)
        N = roles.shape[0]
        lam_birth = params.b * float(N) * dt
        if lam_birth > 0:
            n_birth = int(rng.poisson(lam_birth))
            if n_birth > 0 and N > 0:
                parent_idx = rng.integers(0, N, size=n_birth)
                new_roles = roles[parent_idx].copy()
                new_rel_ids = rel_ids[parent_idx].copy()
                roles = np.concatenate([roles, new_roles], axis=0)
                rel_ids = np.concatenate([rel_ids, new_rel_ids], axis=0)

        # ---------- 7) Mutation on missionaries ----------
        if params.nu is not None:
            idxM = np.where(roles == ROLE_M)[0]
            for idx in idxM:
                r0 = int(rel_ids[idx])
                if r0 == 0 or (r0 not in params.nu):
                    continue
                # total out rate
                out_map = params.nu[r0]
                targets = [l for l in out_map.keys() if l != r0 and out_map[l] > 0]
                if not targets:
                    continue
                total_out = sum(out_map[l] for l in targets)
                p_mut = prob_from_rate(total_out, dt)
                if p_mut > 0 and rng.random() < p_mut:
                    # pick target proportional to nu[r0][l]
                    w = [out_map[l] for l in targets]
                    new_r = int(choose_weighted(targets, w, rng))
                    rel_ids[idx] = new_r
                    # role stays M (matches ODE mutation term on missionaries)

    out = {
        "t": t_series.tolist(),
        "S": S_ts.tolist(),
        "B": {str(r): B_ts[r].tolist() for r in rels},
        "M": {str(r): M_ts[r].tolist() for r in rels},
        "P": {str(r): P_ts[r].tolist() for r in rels},
    }
    return out


def _pick_target_for_S(lambdas: Dict[int, float], rng: np.random.Generator) -> int:
    rels = list(lambdas.keys())
    w = [lambdas[r] for r in rels]
    return int(choose_weighted(rels, w, rng))


def _pick_target_for_r(current_r: int, lambdas: Dict[int, float], rng: np.random.Generator) -> int:
    rels = [r for r in lambdas.keys() if r != current_r]
    w = [lambdas[r] for r in rels]
    return int(choose_weighted(rels, w, rng))
