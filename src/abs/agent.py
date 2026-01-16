"""
Agent role definitions for the ABS model.

Defines integer codes for agent states (S, B, M, P) used in vectorized simulation.
Core module.
"""
from __future__ import annotations

# Role codes for speed (vectorized ABS)
ROLE_S = 0
ROLE_B = 1
ROLE_M = 2
ROLE_P = 3

ROLE_NAME = {
    ROLE_S: "S",
    ROLE_B: "B",
    ROLE_M: "M",
    ROLE_P: "P",
}
