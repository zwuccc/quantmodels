"""Daily adjusted OHLCV. yfinance first, stooq as fallback. Cached as parquet.

One source per ticker for its whole history. Two sources are never spliced.
A cached ticker is never downloaded again unless refresh=True.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from qm.config import data_dir
from qm.data.quality import COLS, check_prices
from qm.data.skiplog import SkipLog

Fetcher = Callable[[str, str], pd.DataFrame]


def fetch_yfinance(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise ValueError("yfinance returned nothing")
    df = df.rename(columns=str.lower)[COLS]
    df.index = pd.DatetimeIndex(df.index.tz_localize(None) if df.index.tz is not None else df.index).normalize()
    df.index.name = "date"
    return df


def fetch_stooq(ticker: str, start: str) -> pd.DataFrame:
    sym = ticker.lower() + ".us"
    r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=30)
    r.raise_for_status()
    if not r.text.startswith("Date"):
        raise ValueError("stooq returned no csv")
    df = pd.read_csv(StringIO(r.text), parse_dates=["Date"]).rename(columns=str.lower).set_index("date")
    df = df[df.index >= pd.Timestamp(start)][COLS]
    df.index.name = "date"
    return df


DEFAULT_FETCHERS: list[tuple[str, Fetcher]] = [("yfinance", fetch_yfinance), ("stooq", fetch_stooq)]


class PriceStore:
    def __init__(self, cfg: dict, fetchers: list[tuple[str, Fetcher]] | None = None,
                 skiplog: SkipLog | None = None, sleep: float | None = None):
        self.cfg = cfg
        self.dir = data_dir(cfg) / "raw" / "prices"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = data_dir(cfg) / "manifest.json"
        self.fetchers = DEFAULT_FETCHERS if fetchers is None else fetchers
        self.skiplog = skiplog or SkipLog(data_dir(cfg) / "logs" / "skipped.csv")
        self.sleep = cfg["prices"]["sleep_between_downloads"] if sleep is None else sleep
        self.network_calls = 0

    def path(self, ticker: str) -> Path:
        return self.dir / f"{ticker}.parquet"

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def _record(self, ticker: str, source: str, rows: int) -> None:
        m = self._manifest()
        m.setdefault("prices", {})[ticker] = {
            "source": source, "rows": rows,
            "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.manifest_path.write_text(json.dumps(m, indent=1, sort_keys=True))

    def get(self, ticker: str, refresh: bool = False) -> pd.DataFrame | None:
        """Cached frame for one ticker, downloading only if not cached."""
        p = self.path(ticker)
        if p.exists() and not refresh:
            return pd.read_parquet(p)
        start = self.cfg["dates"]["data_start"]
        errors = []
        for name, fetch in self.fetchers:
            try:
                self.network_calls += 1
                df = fetch(ticker, start)
                if df is None or df.empty:
                    raise ValueError("empty")
                df = df[COLS].astype(float).sort_index()
                df.to_parquet(p)
                self._record(ticker, name, len(df))
                return df
            except Exception as e:  # noqa: BLE001  any failure moves to the next source
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
            finally:
                if self.sleep:
                    time.sleep(self.sleep)
        self.skiplog.log(ticker, "prices_download", " | ".join(errors))
        return None

    def download_all(self, tickers: list[str], refresh: bool = False, progress_every: int = 50) -> dict:
        ok, failed = 0, 0
        for i, t in enumerate(tickers, 1):
            if self.get(t, refresh=refresh) is None:
                failed += 1
            else:
                ok += 1
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{len(tickers)} done, {failed} failed", flush=True)
        return {"ok": ok, "failed": failed, "network_calls": self.network_calls}


def build_panel(store: PriceStore, tickers: list[str], benchmark: str) -> dict[str, pd.DataFrame]:
    """Wide panels (date x ticker) for open, high, low, close, volume.

    The calendar is the benchmark's trading days. Tickers that fail a quality
    check are dropped and logged. Nothing is filled: a ticker's cells before
    its first day stay NaN.
    """
    bench = store.get(benchmark)
    if bench is None:
        raise RuntimeError(f"benchmark {benchmark} has no data")
    cal = bench.index
    q = store.cfg["data_quality"]
    frames = {}
    for t in tickers:
        df = store.get(t)
        if df is None:
            continue
        df = df[df.index.isin(cal)]
        reason = check_prices(df, cal, q)
        if reason:
            store.skiplog.log(t, "prices_quality", reason)
            continue
        frames[t] = df
    return {c: pd.DataFrame({t: f[c] for t, f in frames.items()}).reindex(cal) for c in COLS}
