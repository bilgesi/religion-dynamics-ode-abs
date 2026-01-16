"""
Runtime configuration for simulations.

Defines default parameters for time stepping, random seeds, output directories,
and ABS replicate counts. Core module used throughout the project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    dt: float = 0.1
    t_max: float = 200.0
    random_seed: int = 42
    abs_replicates: int = 30

    outputs_dir: Path = Path("outputs")
    runs_dir: Path = Path("outputs/runs")
    figs_dir: Path = Path("outputs/figures")


DEFAULT_CONFIG = RunConfig()
