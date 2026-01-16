"""
Export scenario parameters to CSV table.

Extracts all parameters from SCENARIOS dictionary and writes to tabular format.
Analysis module for paper artifact generation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Expecting: SCENARIOS["A"], SCENARIOS["B"], SCENARIOS["C"]
# Each scenario is a function that returns (params, init)
from src.scenarios import SCENARIOS  # type: ignore


def to_dict(x: Any) -> Any:
    return asdict(x) if is_dataclass(x) else x


def fmt(v: Any) -> str:
    """Pretty-print values for table cells."""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, dict):
        # compact dict rendering: if all values equal, show scalar, else show k:v pairs
        try:
            vals = list(v.values())
            if len(vals) > 0 and all(val == vals[0] for val in vals):
                return fmt(vals[0])
        except Exception:
            pass
        items = []
        for k in sorted(v.keys(), key=lambda z: str(z)):
            items.append(f"{k}:{fmt(v[k])}")
        s = "{" + ", ".join(items) + "}"
        return s
    return str(v)


def get_params(scn_key: str) -> Dict[str, Any]:
    scn_func = SCENARIOS[scn_key]
    if not callable(scn_func):
        raise RuntimeError(
            f"SCENARIOS['{scn_key}'] must be a callable that returns (params, init)"
        )
    params, init = scn_func()
    p = to_dict(params)
    return p


def main():
    A = get_params("A")
    B = get_params("B")
    C = get_params("C")

    # Pick a compact set for the main paper table (you can expand later for appendix)
    rows = [
        ("b", "Birth rate", "b"),
        ("mu", "Death rate", "mu"),
        ("beta0", r"Baseline transmission", "beta0"),
        ("q", r"Missionary fraction", "q"),
        ("sigma", r"$B\to M$ rate", "sigma"),
        ("kappa", r"$M\to B$ rate", "kappa"),
        ("tauB", r"$B\to P$ rate", "tauB"),
        ("tauM", r"$M\to P$ rate", "tauM"),
        ("rhoB", r"$B\to S$ (disaffiliation)", "rhoB"),
        ("rhoM", r"$M\to S$ (disaffiliation)", "rhoM"),
        ("rhoP", r"$P\to S$ (disaffiliation)", "rhoP"),
        ("nu", r"Mutation matrix $\nu$", "nu"),
        ("dt", r"ABS step size", "dt"),
        ("t_max", r"Time horizon", "t_max"),
    ]

    out_runs = Path("outputs") / "runs"
    out_runs.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    csv_path = out_runs / "scenario_params_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Description", "Scenario A", "Scenario B", "Scenario C"])
        for key, desc, field in rows:
            w.writerow([key, desc, fmt(A.get(field, "")), fmt(B.get(field, "")), fmt(C.get(field, ""))])

    # --- LaTeX (booktabs) ---
    tex_path = out_runs / "scenario_params_table.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Parameter settings for Scenarios A, B, and C.}\n")
        f.write("\\label{tab:scenario_params}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Description & A & B & C \\\\\n")
        f.write("\\midrule\n")
        for key, desc, field in rows:
            a = fmt(A.get(field, ""))
            b = fmt(B.get(field, ""))
            c = fmt(C.get(field, ""))
            # escape underscores if any show up
            a = a.replace("_", "\\_")
            b = b.replace("_", "\\_")
            c = c.replace("_", "\\_")
            f.write(f"{key} & {desc} & {a} & {b} & {c} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    main()

