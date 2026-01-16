#!/usr/bin/env python3
"""
Generate LaTeX table and updated paragraph for scenario results.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios import scenario_A, scenario_B, scenario_C
from src.config import DEFAULT_CONFIG

# Get parameters
params_A, init_A = scenario_A()
params_B, init_B = scenario_B()
params_C, init_C = scenario_C()

# Milestones (from compute_scenario_milestones.py output)
milestones_A = {
    "takeoff_time": 17.6,
    "crossover_time": 49.9,
    "T": 216.0,
    "y1_final": 0.096,
    "y2_final": 0.890
}

milestones_B = {
    "num_bursts": 5,
    "max_peak_share": 0.254,
    "T": 216.0,
    "minors_final_total_share": 0.032
}

milestones_C = {
    "y1_equilibrium": 0.936,
    "y2_equilibrium": 0.009,
    "T": 400.0,
    "settling_time": 192.6
}

# LaTeX table
latex_table = f"""
\\begin{{table}}[!ht]
  \\centering
  \\caption{{Parameter sets and initial conditions for stylized scenarios.}}
  \\label{{tab:scenario_params}}
  \\begin{{tabular}}{{lccc}}
    \\toprule
    Parameter & Scenario A & Scenario B & Scenario C \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Demography}}}} \\\\
    $b$ & {params_A.b:.3f} & {params_B.b:.3f} & {params_C.b:.3f} \\\\
    $\\mu$ & {params_A.mu:.3f} & {params_B.mu:.3f} & {params_C.mu:.3f} \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Religion 1}}}} \\\\
    $\\beta_0$ & {params_A.beta0[1]:.3f} & {params_B.beta0[1]:.3f} & {params_C.beta0[1]:.3f} \\\\
    $q$ & {params_A.q[1]:.3f} & {params_B.q[1]:.3f} & {params_C.q[1]:.3f} \\\\
    $\\sigma$ & {params_A.sigma[1]:.3f} & {params_B.sigma[1]:.3f} & {params_C.sigma[1]:.3f} \\\\
    $\\kappa$ & {params_A.kappa[1]:.3f} & {params_B.kappa[1]:.3f} & {params_C.kappa[1]:.3f} \\\\
    $\\rho_B$ & {params_A.rhoB[1]:.3f} & {params_B.rhoB[1]:.3f} & {params_C.rhoB[1]:.3f} \\\\
    $\\rho_M$ & {params_A.rhoM[1]:.3f} & {params_B.rhoM[1]:.3f} & {params_C.rhoM[1]:.3f} \\\\
    $\\rho_P$ & {params_A.rhoP[1]:.3f} & {params_B.rhoP[1]:.3f} & {params_C.rhoP[1]:.3f} \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Religion 2}}}} \\\\
    $\\beta_0$ & {params_A.beta0[2]:.3f} & {params_B.beta0[2]:.3f} & {params_C.beta0[2]:.3f} \\\\
    $q$ & {params_A.q[2]:.3f} & {params_B.q[2]:.3f} & {params_C.q[2]:.3f} \\\\
    $\\sigma$ & {params_A.sigma[2]:.3f} & {params_B.sigma[2]:.3f} & {params_C.sigma[2]:.3f} \\\\
    $\\kappa$ & {params_A.kappa[2]:.3f} & {params_B.kappa[2]:.3f} & {params_C.kappa[2]:.3f} \\\\
    $\\rho_B$ & {params_A.rhoB[2]:.3f} & {params_B.rhoB[2]:.3f} & {params_C.rhoB[2]:.3f} \\\\
    $\\rho_M$ & {params_A.rhoM[2]:.3f} & {params_B.rhoM[2]:.3f} & {params_C.rhoM[2]:.3f} \\\\
    $\\rho_P$ & {params_A.rhoP[2]:.3f} & {params_B.rhoP[2]:.3f} & {params_C.rhoP[2]:.3f} \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Mutation}}}} \\\\
    $\\nu_{{1\\to 2}}$ & {params_A.nu[1][2]:.3f} & 0.000 & {params_C.nu[1][2]:.3f} \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Initial conditions}}}} \\\\
    $S(0)$ & {init_A['S0']} & {init_B['S0']} & {init_C['S0']} \\\\
    $B_1(0)$ & {init_A['B0'][1]} & {init_B['B0'][1]} & {init_C['B0'][1]} \\\\
    $M_1(0)$ & {init_A['M0'][1]} & {init_B['M0'][1]} & {init_C['M0'][1]} \\\\
    $P_1(0)$ & {init_A['P0'][1]} & {init_B['P0'][1]} & {init_C['P0'][1]} \\\\
    $B_2(0)$ & {init_A['B0'][2]} & {init_B['B0'][2]} & {init_C['B0'][2]} \\\\
    $M_2(0)$ & {init_A['M0'][2]} & {init_B['M0'][2]} & {init_C['M0'][2]} \\\\
    $P_2(0)$ & {init_A['P0'][2]} & {init_B['P0'][2]} & {init_C['P0'][2]} \\\\
    \\midrule
    \\multicolumn{{4}}{{l}}{{\\textit{{Simulation}}}} \\\\
    $T$ & {params_A.t_max:.1f} & {params_B.t_max:.1f} & {params_C.t_max:.1f} \\\\
    $\\Delta t$ & {params_A.dt:.1f} & {params_B.dt:.1f} & {params_C.dt:.1f} \\\\
    Replicates & {DEFAULT_CONFIG.abs_replicates} & {DEFAULT_CONFIG.abs_replicates} & {DEFAULT_CONFIG.abs_replicates} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

# Updated paragraph
updated_paragraph = f"""
Fig.~\\ref{{fig:new_religion_threepanel}} presents replicate-mean time-series of population shares under the three stylized parameterizations (Scenarios~A-C), illustrating how the same mechanistic model produces qualitatively distinct macrodynamics when only parameters and initial conditions are changed. Panel (a) shows the emergence of a new religion (strain) that begins at (near) zero prevalence and then grows to dominate the population; panel (b) shows repeated low-prevalence bursts of multiple minor religions that fail to persist; and panel (c) shows a regime of stable coexistence in which two religions converge to a near-stationary composition after an initial transient. In particular, in Fig.~\\ref{{fig:new_religion_threepanel}}a, we observe a characteristic \\say{{establishment-from-rarity}} trajectory: religion~2 remains near zero for an initial period (driven by mutation-mediated seeding), then passes an apparent takeoff point where growth accelerates, followed by saturation as the susceptible pool is depleted and competitive conversion intensifies. The incumbent (religion~1) declines as the challenger rises, with the crossover marking a replacement-like transition. Quantitatively, religion~2 crosses the 1\\% threshold at $t = {milestones_A['takeoff_time']:.1f}$ weeks, achieves parity with religion~1 at $t = {milestones_A['crossover_time']:.1f}$ weeks, and reaches final shares of $y_1 = {milestones_A['y1_final']:.3f}$ and $y_2 = {milestones_A['y2_final']:.3f}$ at $T = {milestones_A['T']:.0f}$ weeks. In Fig.~\\ref{{fig:new_religion_threepanel}}b, the dominant religion remains relatively stable while several minor strains appear, peak, and then shrink, producing a sequence of transient \\say{{cult-like}} waves; importantly, these waves remain bounded and do not accumulate into long-run dominance. The simulation produces {milestones_B['num_bursts']} distinct bursts, with the largest minor strain reaching a peak share of {milestones_B['max_peak_share']:.3f}, while the combined final share of all minor strains at $T = {milestones_B['T']:.0f}$ weeks is only {milestones_B['minors_final_total_share']:.3f}. In Fig.~\\ref{{fig:new_religion_threepanel}}c, both religions initially adjust due to the mismatch between initial conditions and the long-run attractor, after which the shares flatten and remain approximately constant over the extended horizon, indicating stable pluralism. The system settles to equilibrium shares of $y_1^* = {milestones_C['y1_equilibrium']:.3f}$ and $y_2^* = {milestones_C['y2_equilibrium']:.3f}$ (mean over the final 20\\% of the {milestones_C['T']:.0f}-week horizon), with convergence to within $\\pm 0.01$ of these values achieved by $t = {milestones_C['settling_time']:.1f}$ weeks. Exact parameter values and initial conditions are provided in Table~\\ref{{tab:scenario_params}}.
"""

# Fixed captions
fixed_captions = """
% (a)
\\begin{subfigure}[t]{0.32\\textwidth}
  \\centering
  \\includegraphics[width=\\linewidth]{fig_scenario_A.pdf}
  \\caption{Emergence and replacement of the old religion.}
  \\label{fig:new_religion_timeseries_a}
\\end{subfigure}\\hfill
% (b)
\\begin{subfigure}[t]{0.32\\textwidth}
  \\centering
  \\includegraphics[width=\\linewidth]{fig_scenario_B.pdf}
  \\caption{Transient cult-like waves.}
  \\label{fig:new_religion_sensitivity_b}
\\end{subfigure}\\hfill
% (c)
\\begin{subfigure}[t]{0.32\\textwidth}
  \\centering
  \\includegraphics[width=\\linewidth]{fig_scenario_C.pdf}
  \\caption{Stable coexistence.}
  \\label{fig:scenarioC_timeseries_c}
\\end{subfigure}
"""

print("=" * 70)
print("LATEX OUTPUT")
print("=" * 70)
print("\n1. UPDATED PARAGRAPH:")
print("-" * 70)
print(updated_paragraph)
print("\n2. LATEX TABLE:")
print("-" * 70)
print(latex_table)
print("\n3. FIXED CAPTIONS:")
print("-" * 70)
print(fixed_captions)
print("=" * 70)
