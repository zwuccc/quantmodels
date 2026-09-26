"""SEC EDGAR: ticker to CIK map and XBRL companyfacts, cached and rate limited."""
from __future__ import annotations

import gzip
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Callable

import requests

from qm.config import data_dir
from qm.data.skiplog import SkipLog


class RateLimiter:
    """At most max_calls in any rolling 1 second window."""

    def __init__(self, max_calls: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_calls = max_calls
        self.clock = clock
        self.sleep = sleep
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self.calls and now - self.calls[0] >= 1.0:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            self.sleep(1.0 - (now - self.calls[0]) + 1e-3)
            now = self.clock()
            while self.calls and now - self.calls[0] >= 1.0:
                self.calls.popleft()
        self.calls.append(now)


def user_agent(cfg: dict) -> str:
    ua = os.environ.get(cfg["edgar"]["user_agent_env"], "").strip()
    if "@" not in ua:
        raise RuntimeError(
            f"Set {cfg['edgar']['user_agent_env']} to a name and email, "
            'e.g. export QM_SEC_USER_AGENT="Jane Doe jane@example.com". SEC requires it.')
    return ua


class EdgarClient:
    def __init__(self, cfg: dict, session: requests.Session | None = None,
                 limiter: RateLimiter | None = None, ua: str | None = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.limiter = limiter or RateLimiter(cfg["edgar"]["max_requests_per_sec"])
        self.headers = {"User-Agent": ua or user_agent(cfg), "Accept-Encoding": "gzip, deflate"}
        self.dir = data_dir(cfg) / "raw" / "edgar"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skiplog = SkipLog(data_dir(cfg) / "logs" / "skipped.csv")

    def get_json(self, url: str) -> dict:
        self.limiter.wait()
        r = self.session.get(url, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def ticker_map(self, refresh: bool = False) -> dict[str, int]:
        p = self.dir / "company_tickers.json"
        if p.exists() and not refresh:
            raw = json.loads(p.read_text())
        else:
            raw = self.get_json(self.cfg["edgar"]["tickers_url"])
            p.write_text(json.dumps(raw))
        return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}

    def facts_path(self, cik: int) -> Path:
        return self.dir / f"CIK{cik:010d}.json.gz"

    def companyfacts(self, cik: int, refresh: bool = False) -> dict | None:
        p = self.facts_path(cik)
        if p.exists() and not refresh:
            with gzip.open(p, "rt") as f:
                return json.load(f)
        try:
            data = self.get_json(self.cfg["edgar"]["facts_url"].format(cik=cik))
        except Exception as e:  # noqa: BLE001
            self.skiplog.log(f"CIK{cik}", "edgar_download", f"{type(e).__name__}: {str(e)[:80]}")
            return None
        with gzip.open(p, "wt") as f:
            json.dump(data, f)
        return data


def load_cached_facts(cfg: dict, cik: int) -> dict | None:
    p = data_dir(cfg) / "raw" / "edgar" / f"CIK{cik:010d}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)
