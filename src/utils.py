"""
Utility functions for file I/O and directory management.

Provides helpers for JSON serialization and directory creation.
Core module used by experiments and analysis scripts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
