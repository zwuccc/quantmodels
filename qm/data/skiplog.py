"""Rule 8: missing or bad data means skip the name and log it. Never fill."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["time", "name", "stage", "reason"]


class SkipLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, stage: str, reason: str) -> None:
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "name": name, "stage": stage, "reason": reason,
            })


def skiplog_for(cfg: dict) -> SkipLog:
    from qm.config import data_dir
    return SkipLog(data_dir(cfg) / "logs" / "skipped.csv")
