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
from qm.data.quality import COLS, check_adjustments, check_prices, trim_leading_stale
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


UNADJ_COLS = ["close_raw", "adj_close", "dividends", "splits"]


def fetch_yfinance_unadjusted(ticker: str, start: str) -> pd.DataFrame:
    """Split adjusted but NOT dividend adjusted close, Yahoo's adjusted close, and
    the corporate actions. Used only to audit Yahoo's adjustments."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, auto_adjust=False, actions=True)
    if df is None or df.empty:
        raise ValueError("yfinance returned nothing")
    df = df.rename(columns={"Close": "close_raw", "Adj Close": "adj_close",
                            "Dividends": "dividends", "Stock Splits": "splits"})[UNADJ_COLS]
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
                 skiplog: SkipLog | None = None, sleep: float | None = None,
                 subdir: str = "prices", cols: list[str] | None = None):
        self.cfg = cfg
        self.subdir = subdir
        self.cols = cols or COLS
        self.dir = data_dir(cfg) / "raw" / subdir
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
        m.setdefault(self.subdir, {})[ticker] = {
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
                df = df[self.cols].astype(float).sort_index()
                df.to_parquet(p)
                self._record(ticker, name, len(df))
                return df
            except Exception as e:  # noqa: BLE001  any failure moves to the next source
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
            finally:
                if self.sleep:
                    time.sleep(self.sleep)
        self.skiplog.log(ticker, f"{self.subdir}_download", " | ".join(errors))
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


def build_panel(store: PriceStore, tickers: list[str], benchmark: str,
                audit: PriceStore | None = None) -> dict[str, pd.DataFrame]:
    """Wide panels (date x ticker) for open, high, low, close, volume.

    The calendar is the benchmark's trading days. Days Yahoo filled with a
    copied price count as missing. A fake history before real trading starts
    is cut off. Tickers that fail a check are dropped and logged. Nothing is
    filled: a ticker's cells before its first day stay NaN. With an audit store
    (unadjusted prices and corporate actions), broken adjustments are caught too.
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
        df, filled, cut = trim_leading_stale(df, q["min_clean_run"])
        if cut:
            store.skiplog.log(t, "prices_trimmed", f"cut {cut} leading days of copied prices before real trading")
        reason = None
        if filled > q["max_internal_gaps"]:
            reason = f"{filled} filled days (zero volume, copied close) inside real trading"
        reason = reason or check_prices(df, cal, q)
        if reason is None and audit is not None:
            reason = check_adjustments(df, audit.get(t), q)
        if reason:
            store.skiplog.log(t, "prices_quality", reason)
            continue
        frames[t] = df
    return {c: pd.DataFrame({t: f[c] for t, f in frames.items()}).reindex(cal) for c in COLS}
