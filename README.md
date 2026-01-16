# religion-dynamics-ode-abs

A computational framework for modeling religious dynamics using both Ordinary Differential Equations (ODE) and Agent-Based Simulation (ABS) approaches.

## Overview

This project implements a hybrid model to simulate the spread and decline of religious groups in populations. It combines:

- **ODE Model**: Deterministic mean-field dynamics for rapid scenario exploration
- **ABS Model**: Stochastic agent-based simulation for detailed behavior modeling

## Project Structure

```
abs-religious-spread-model/
├── src/                    # Core model implementation
│   ├── abs/                # Agent-based simulation
│   ├── ode/                # ODE solver and system
│   ├── model/              # Model parameters and rates
│   ├── config.py           # Run configuration
│   ├── scenarios.py        # Predefined scenarios (A, B, C)
│   └── utils.py            # Utility functions
│
├── experiments/            # Experiment scripts
│   ├── scenarios/          # Scenario runners (A, B, C)
│   ├── sweeps/             # Parameter sweep experiments
│   ├── heatmaps/           # Phase transition heatmaps
│   ├── historical_fit/     # Historical data fitting
│   └── tests/              # Test and validation scripts
│
├── analysis/               # Analysis and visualization
│
├── data/                   # Data files
│   ├── raw/                # Raw data by country (nz, finland, sweden, jw)
│   └── processed/          # Processed time series
│
├── outputs/                # Generated outputs
│   ├── figures/            # PDF/PNG figures
│   ├── tables/             # LaTeX tables
│   └── runs/               # Run artifacts (JSON, CSV)
│
├── paper/                  # Paper-related files
│   └── data/               # Data for paper figures
│
├── scripts/                # Utility scripts
├── logs/                   # Log files
└── _legacy/                # Legacy/cache files
```

## Usage

All scripts can be run from the project root directory:

```bash
# Run scenario experiments
python experiments/scenarios/run_scenarios.py

# Generate heatmap (full: 10x10 grid, ~300 ABS runs, takes longer)
python experiments/heatmaps/heatmap_abc_abs.py

# Quick heatmap test (3x3 grid, 1 repeat, ~5 minutes)
python experiments/heatmaps/heatmap_abc_abs.py --nx 3 --ny 3 --R 1 --T 50 --dt 0.5

# Historical fitting
python experiments/historical_fit/fig_historic_fit.py
```

## Key Scenarios

- **Scenario A**: Dominant religion replacement
- **Scenario B**: Sequential strain invasions with seeding
- **Scenario C**: Transient minority dynamics

## Dependencies

- Python 3.11+
- numpy, scipy, matplotlib, pandas
