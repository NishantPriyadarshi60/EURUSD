"""Thin YAML config loader. Every parameter used anywhere in this project must come
through here (or be read directly from one of these files) — nothing hardcoded (spec §9).
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load a config file by stem, e.g. load('data') -> configs/data.yaml contents."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No such config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
