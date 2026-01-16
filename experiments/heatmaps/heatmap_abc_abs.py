# experiments/heatmap_abc_abs.py
"""
Phase-transition heatmap: 3 regimes (A/B/C) as function of (beta0_2, rhoM_2).
Scenario A setup, r=2.

Defaults:
- 10x10 grid, R=3 repeats => 300 total runs
- T=216 weeks, dt=0.1
- Classification: A>=0.30, C in [0.02,0.30) with stab<=0.005, else B
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Any, List
from multiprocessing import Pool, cpu_count

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.scenarios import SCENARIOS
from src.abs.sim_abs import run_abs
from src.abs.init_population import init_population_from_counts
from src.model.params import ModelParams, copy_with_dt


# ----------------------------
# Globals (set once per worker)
# ----------------------------
_G_PARAMS0: ModelParams | None = None
_G_INIT_COUNTS: Dict[str, Any] | None = None
_G_REL_ID: int | None = None
_G_TIE_RHOB: bool = False
_G_DT: float = 0.1
_G_TMAX: float = 216.0
_G_SEED: int = 12345
_G_R: int = 3
_G_THR_A: float = 0.30
_G_THR_C: float = 0.02
_G_STAB_THR: float = 0.005


def _get_case(key_hint: str = "A") -> Tuple[ModelParams, Dict]:
    """Get (params, init) from SCENARIOS."""
    keys_to_try = [key_hint, key_hint.upper(), key_hint.lower()]
    key = None
    for k in keys_to_try:
        if k in SCENARIOS:
            key = k
            break
    if key is None:
        key = list(SCENARIOS.keys())[0]
    return SCENARIOS[key]()


def _get_series(d: Dict, r: int) -> np.ndarray:
    """Allow dict keys to be int or str."""
    if r in d:
        return np.asarray(d[r], dtype=float)
    if str(r) in d:
        return np.asarray(d[str(r)], dtype=float)
    raise KeyError(f"Missing religion {r} in trajectory dict keys={list(d.keys())[:5]}...")


def _extract_y(traj_out: Dict, rel_id: int) -> np.ndarray:
    """Extract y_r(t) = (B_r + M_r + P_r)/N from ABS output."""
    S = np.asarray(traj_out["S"], dtype=float)
    N = S.copy()

    # religions inferred from keys in B
    b_keys = traj_out["B"].keys()
    religions: List[int] = []
    for k in b_keys:
        try:
            r = int(k)
            if r != 0:  # Filter out 0/none religion if present
                religions.append(r)
        except Exception:
            pass
    religions = sorted(religions)

    for r in religions:
        Br = _get_series(traj_out["B"], r)
        Mr = _get_series(traj_out["M"], r)
        Pr = _get_series(traj_out["P"], r)
        N += Br + Mr + Pr

    Br = _get_series(traj_out["B"], rel_id)
    Mr = _get_series(traj_out["M"], rel_id)
    Pr = _get_series(traj_out["P"], rel_id)
    return (Br + Mr + Pr) / np.clip(N, 1e-12, None)


def classify_abc(
    y: np.ndarray,
    last_frac: float = 0.2,
    thr_A: float = 0.30,
    thr_C: float = 0.02,
    stab_thr: float = 0.005,
) -> Tuple[str, float, float, float]:
    """Return (label, y_final, y_max, stability)."""
    T = len(y)
    k = max(5, int(T * last_frac))
    tail = y[-k:]
    y_final = float(np.mean(tail))
    y_max = float(np.max(y))
    stab = float(np.std(tail))
    if y_final >= thr_A:
        return "A", y_final, y_max, stab
    if (y_final >= thr_C) and (y_final < thr_A) and (stab <= stab_thr):
        return "C", y_final, y_max, stab
    return "B", y_final, y_max, stab


def _init_worker(
    params0: ModelParams,
    init0: Dict,
    rel_id: int,
    tie_rhoB: bool,
    dt: float,
    t_max: float,
    seed: int,
    R: int,
    thr_A: float,
    thr_C: float,
    stab_thr: float,
):
    """Run once per worker. Stores shared data in globals to avoid per-task pickling."""
    global _G_PARAMS0, _G_INIT_COUNTS, _G_REL_ID, _G_TIE_RHOB, _G_DT, _G_TMAX, _G_SEED, _G_R
    global _G_THR_A, _G_THR_C, _G_STAB_THR

    _G_PARAMS0 = params0
    _G_REL_ID = rel_id
    _G_TIE_RHOB = tie_rhoB
    _G_DT = dt
    _G_TMAX = t_max
    _G_SEED = seed
    _G_R = R
    _G_THR_A = thr_A
    _G_THR_C = thr_C
    _G_STAB_THR = stab_thr

    # Pre-parse counts once
    S0 = int(init0["S0"])
    B0 = {int(k): int(v) for k, v in init0["B0"].items()}
    M0 = {int(k): int(v) for k, v in init0["M0"].items()}
    P0 = {int(k): int(v) for k, v in init0["P0"].items()}
    _G_INIT_COUNTS = {"S0": S0, "B0": B0, "M0": M0, "P0": P0}


def run_single_cell(cell: Tuple[int, int, float, float]) -> Tuple[int, int, int, Dict]:
    """Compute one grid cell."""
    iy, ix, beta, rho = cell
    assert _G_PARAMS0 is not None and _G_INIT_COUNTS is not None and _G_REL_ID is not None

    rel_id = _G_REL_ID

    # Fresh params per cell (copy_with_dt already copies all dicts, so they're safe to mutate)
    params = copy_with_dt(_G_PARAMS0, dt=_G_DT, t_max=_G_TMAX)

    # Modify beta0 and rhoM for this cell's parameter values
    params.beta0[rel_id] = float(beta)
    params.rhoM[rel_id] = float(rho)
    if _G_TIE_RHOB:
        params.rhoB[rel_id] = float(rho)

    S0 = _G_INIT_COUNTS["S0"]
    B0_base = _G_INIT_COUNTS["B0"]
    M0_base = _G_INIT_COUNTS["M0"]
    P0_base = _G_INIT_COUNTS["P0"]

    ys = []
    for rep in range(_G_R):
        rep_seed = _G_SEED + 100000 * iy + 1000 * ix + rep
        rng_rep = np.random.default_rng(rep_seed)

        # IMPORTANT correctness: pass fresh dict copies in case init_population_from_counts mutates
        roles, rel_ids = init_population_from_counts(
            S0,
            dict(B0_base),
            dict(M0_base),
            dict(P0_base),
            params.religions,
            rng_rep,
        )
        out = run_abs(params, roles, rel_ids, seed=rep_seed)
        ys.append(_extract_y(out, rel_id=rel_id))

    y_mean = np.mean(np.stack(ys, axis=0), axis=0)
    label, y_final, y_max, stab = classify_abc(
        y_mean, thr_A=_G_THR_A, thr_C=_G_THR_C, stab_thr=_G_STAB_THR
    )
    lab_id = {"B": 0, "C": 1, "A": 2}[label]

    stats_row = {
        "beta0_rel": float(beta),
        "rhoM_rel": float(rho),
        "label": label,
        "y_final": y_final,
        "y_max": y_max,
        "stability": stab,
        "R": _G_R,
        "dt": _G_DT,
        "T": _G_TMAX,
    }
    
    return iy, ix, lab_id, stats_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A")
    ap.add_argument("--rel", type=int, default=2)

    # default 10x10
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=10)

    ap.add_argument("--beta_min", type=float, default=0.05)
    ap.add_argument("--beta_max", type=float, default=0.60)
    ap.add_argument("--rho_min", type=float, default=0.0005)
    ap.add_argument("--rho_max", type=float, default=0.02)

    ap.add_argument("--tie_rhoB", action="store_true")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=216.0)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--thr_A", type=float, default=0.30)
    ap.add_argument("--thr_C", type=float, default=0.02)
    ap.add_argument("--stab_thr", type=float, default=0.005)

    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=6)

    ap.add_argument("--outdir", default="outputs/heatmap_phase")

    args = ap.parse_args()

    params0, init0 = _get_case(args.case)

    # ensure rel exists in params and init (Scenario A already has r=2)
    if args.rel not in params0.religions:
        params0.religions.append(args.rel)
        params0.beta0[args.rel] = 0.3
        params0.q[args.rel] = 0.05
        params0.sigma[args.rel] = 0.02
        params0.kappa[args.rel] = 0.02
        params0.tauB[args.rel] = 0.002
        params0.tauM[args.rel] = 0.003
        params0.rhoB[args.rel] = 0.006
        params0.rhoM[args.rel] = 0.006
        params0.rhoP[args.rel] = 0.001
        if params0.nu is not None:
            params0.nu[args.rel] = {}
    
    # ensure rel exists in init counts
    if args.rel not in init0["B0"]:
        init0["B0"][args.rel] = 0
        init0["M0"][args.rel] = 0
        init0["P0"][args.rel] = 0

    # build grids
    beta_grid = np.linspace(args.beta_min, args.beta_max, args.nx)
    rho_grid = np.linspace(args.rho_min, args.rho_max, args.ny)

    total_cells = args.nx * args.ny
    total_sims = total_cells * args.R

    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()

    print(f"Grid: {args.nx}x{args.ny} = {total_cells} cells | R={args.R} => {total_sims} ABS runs")
    print(f"Using {n_jobs} workers | chunksize={args.chunksize}")

    cell_list: List[Tuple[int, int, float, float]] = []
    for iy, rho in enumerate(rho_grid):
        for ix, beta in enumerate(beta_grid):
            cell_list.append((iy, ix, float(beta), float(rho)))

    label_grid = np.zeros((args.ny, args.nx), dtype=int)
    stats_rows: List[Dict[str, Any]] = []

    start = time.time()
    done = 0

    with Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(
            params0,
            init0,
            args.rel,
            args.tie_rhoB,
            args.dt,
            args.T,
            args.seed,
            args.R,
            args.thr_A,
            args.thr_C,
            args.stab_thr,
        ),
    ) as pool:
        for iy, ix, lab_id, row in pool.imap_unordered(
            run_single_cell, cell_list, chunksize=max(1, args.chunksize)
        ):
            label_grid[iy, ix] = lab_id
            stats_rows.append(row)
            done += 1

            # Print progress periodically (every 10% or at start/end)
            if done == 1 or done == total_cells or done % max(1, total_cells // 10) == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total_cells - done) / rate if rate > 0 else 0
                print(f"Progress: {done}/{total_cells} ({100*done/total_cells:.1f}%) | "
                      f"{rate:.3f} cells/s | ETA: {remaining/60:.1f} min", flush=True)

    stats_rows.sort(key=lambda r: (r["rhoM_rel"], r["beta0_rel"]))

    out_base = Path(args.outdir)
    figs_dir = out_base / "figs"
    runs_dir = out_base / "runs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # save CSV
    csv_path = runs_dir / "heatmap_phase_transition.csv"
    cols = ["beta0_2", "rhoM_2", "label", "y_final", "stab"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in stats_rows:
            f.write(f"{row['beta0_rel']},{row['rhoM_rel']},{row['label']},{row['y_final']},{row['stability']}\n")
    print(f"Saved CSV: {csv_path}")

    # Compute percentages
    labels = [r["label"] for r in stats_rows]
    n_A = labels.count("A")
    n_B = labels.count("B")
    n_C = labels.count("C")
    total = len(labels)
    print(f"\n=== Regime Distribution ===")
    print(f"A (takeover/large):     {n_A}/{total} = {100*n_A/total:.1f}%")
    print(f"B (small/transient):    {n_B}/{total} = {100*n_B/total:.1f}%")
    print(f"C (stable coexistence): {n_C}/{total} = {100*n_C/total:.1f}%")

    # plot heatmap
    cmap = ListedColormap(["#d9d9d9", "#9ecae1", "#fdae6b"])  # B, C, A
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8, 6))
    extent = [beta_grid[0], beta_grid[-1], rho_grid[0], rho_grid[-1]]
    plt.imshow(label_grid, origin="lower", aspect="auto", cmap=cmap, norm=norm, extent=extent)
    plt.xlabel(r"$\beta_{0,2}$ (baseline conversion strength)")
    plt.ylabel(r"$\rho^{M}_{2}$ (missionary disaffiliation rate)")
    plt.title("Phase Transition: Scenario A (r=2)")

    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["B: small/transient", "C: stable coexist", "A: takeover"])

    plt.tight_layout()

    pdf_path = figs_dir / "fig_phase_transition_heatmap.pdf"
    png_path = figs_dir / "fig_phase_transition_heatmap.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
