"""Load config/default.yaml into a plain dict, with a few helpers."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None, synthetic: bool = False) -> dict:
    """Read the yaml config.

    synthetic=True points data and results at synthetic/ so a demo run on fake
    data can never touch the real cache, trials log or holdout lock.
    """
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["synthetic"] = synthetic
    if synthetic:
        cfg["paths"] = {"data_dir": "synthetic/data", "results_dir": "synthetic/results"}
    for key in ("data_dir", "results_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def data_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"])


def results_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_dir"])


def cost_rate(cfg: dict) -> float:
    """Total cost per side as a fraction of traded value."""
    c = cfg["costs"]
    return (c["commission_bps"] + c["slippage_bps"]) / 1e4


def strategy_params(cfg: dict, name: str, overrides: dict | None = None) -> dict:
    params = copy.deepcopy(cfg["strategies"][name])
    if overrides:
        params.update(overrides)
    return params
