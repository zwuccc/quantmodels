"""Rule 4: every parameter combination tried is written to trials.csv."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["trial_id", "time", "git_sha", "phase", "strategy", "params", "period",
          "start", "end", "costs_on", "days", "sharpe", "sharpe_daily", "skew", "kurt",
          "cagr", "max_dd"]


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


class TrialLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sha = _sha()

    def log(self, phase: str, strategy: str, params: dict, period: str, costs_on: bool, m: dict) -> None:
        new = not self.path.exists()
        n = 0 if new else sum(1 for _ in open(self.path)) - 1
        row = {"trial_id": n + 1, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_sha": self.sha, "phase": phase, "strategy": strategy,
               "params": params_key(params), "period": period, "start": m["start"], "end": m["end"],
               "costs_on": costs_on, **{k: m[k] for k in ("days", "sharpe", "sharpe_daily", "skew", "kurt", "cagr", "max_dd")}}
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=FIELDS)
        return pd.read_csv(self.path)

    def n_trials(self, strategy: str | None = None) -> int:
        """Distinct (strategy, params) combos. Costs on/off of one combo count once."""
        df = self.read()
        if strategy is not None:
            df = df[df["strategy"].astype(str).str.split("@").str[0] == strategy]
        return int(df[["strategy", "params"]].drop_duplicates().shape[0])

    def sharpe_variance(self, strategy: str | None = None) -> float:
        """Variance of daily Sharpe across distinct combos, measured on the dev period."""
        df = self.read()
        df = df[(df["costs_on"].astype(str) == "True")]
        if strategy is not None:
            df = df[df["strategy"].astype(str).str.split("@").str[0] == strategy]
        dev = df[df["period"] == "dev"]
        use = dev if dev[["strategy", "params"]].drop_duplicates().shape[0] >= 2 else df
        per = use.groupby(["strategy", "params"])["sharpe_daily"].mean()
        return float(np.var(per, ddof=1)) if len(per) >= 2 else 0.0
