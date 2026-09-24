"""Time splits and the holdout lock (rule 3)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qm.config import results_dir


class HoldoutLocked(RuntimeError):
    pass


@dataclass
class Splits:
    data_start: pd.Timestamp
    insample_end: pd.Timestamp
    holdout_start: pd.Timestamp
    snapshot_end: pd.Timestamp
    wf_first_test_year: int

    @property
    def dev_end(self) -> pd.Timestamp:
        """Last day anyone may look at before Phase 5."""
        return self.holdout_start - pd.Timedelta(days=1)

    def wf_windows(self) -> list[dict]:
        """Expanding train window from data_start, one test year at a time."""
        out = []
        y = self.wf_first_test_year
        while pd.Timestamp(f"{y}-01-01") <= self.dev_end:
            test_start = pd.Timestamp(f"{y}-01-01")
            test_end = min(pd.Timestamp(f"{y}-12-31"), self.dev_end)
            out.append({"label": str(y), "train_start": self.data_start,
                        "train_end": test_start - pd.Timedelta(days=1),
                        "test_start": test_start, "test_end": test_end})
            y += 1
        return out


def make_splits(cfg: dict, calendar: pd.DatetimeIndex) -> Splits:
    d = cfg["dates"]
    snap = pd.Timestamp(d["snapshot_end"]) if d.get("snapshot_end") else calendar[-1]
    cut = snap - pd.DateOffset(years=d["holdout_years"])
    after = calendar[calendar > cut]
    return Splits(pd.Timestamp(d["data_start"]), pd.Timestamp(d["insample_end"]),
                  after[0], snap, int(d["wf_first_test_year"]))


def holdout_record_path(cfg: dict) -> Path:
    return results_dir(cfg) / "holdout_record.json"


def holdout_used(cfg: dict) -> bool:
    return holdout_record_path(cfg).exists()


def claim_holdout(cfg: dict, settings: dict) -> None:
    """Called once, right before the Phase 5 run. A second call is refused."""
    p = holdout_record_path(cfg)
    if p.exists():
        raise HoldoutLocked(f"holdout already used, see {p}. It can only be touched once.")
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = ""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "git_sha": sha, "settings": settings}, indent=1, default=str))
