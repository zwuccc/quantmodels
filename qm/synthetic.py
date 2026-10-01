"""A fake market for exercising the whole pipeline without the network.

Prices are random walks with a common market factor. Earnings and balance
sheet items are random too, and go through the same XBRL parsing code as the
real SEC data. By design no strategy has an edge here (unless plant= is set),
so this is a null test: if something looks great on noise, there's a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qm.config import data_dir
from qm.data.fundamentals import build_pit, extract_rows

SECTORS = ["Information Technology", "Health Care", "Industrials", "Consumer Discretionary",
           "Energy", "Financials", "Real Estate", "Materials"]


def _prices(rng, dates, mu, vol, beta, mkt):
    """Log returns. The -vol^2/2 term makes the expected SIMPLE return of the
    stock's own noise zero, so equal weighting noisy names earns no free bonus."""
    n = len(dates)
    r = mu / 252 - vol ** 2 / 2 / 252 + beta * mkt + rng.normal(0, vol / np.sqrt(252), n)
    close = 50 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.004, n))
    rng_ = np.abs(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * (1 + rng_)
    low = np.minimum(open_, close) * (1 - rng_)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1e6}, index=dates)


def _facts(rng, ticker, first_year, last_date, has_cogs):
    eps, rev, cogs, assets = [], [], [], []
    level, a = rng.uniform(0.5, 3), rng.uniform(1e9, 5e10)
    for y in range(first_year, last_date.year + 1):
        qs = []
        for q, (s, e) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30")]):
            end = pd.Timestamp(f"{y}-{e}")
            filed = end + pd.Timedelta(days=int(rng.integers(25, 45)))
            level += rng.normal(0, 0.08)
            v = round(level + rng.normal(0, 0.1), 2)
            qs.append(v)
            if filed <= last_date:
                eps.append({"start": f"{y}-{s}", "end": end.date().isoformat(), "val": v, "form": "10-Q",
                            "filed": filed.date().isoformat(), "accn": f"{ticker}{y}q{q}"})
        end = pd.Timestamp(f"{y}-12-31")
        filed = end + pd.Timedelta(days=int(rng.integers(45, 75)))
        if filed > last_date:
            continue
        q4 = round(level + rng.normal(0, 0.1), 2)
        k = {"form": "10-K", "filed": filed.date().isoformat(), "accn": f"{ticker}{y}k"}
        eps.append({"start": f"{y}-01-01", "end": end.date().isoformat(), "val": round(sum(qs) + q4, 2), **k})
        a *= np.exp(rng.normal(0.05, 0.1))
        r_ = a * rng.uniform(0.3, 1.5)
        rev.append({"start": f"{y}-01-01", "end": end.date().isoformat(), "val": r_, **k})
        cogs.append({"start": f"{y}-01-01", "end": end.date().isoformat(), "val": r_ * rng.uniform(0.3, 0.9), **k})
        assets.append({"end": end.date().isoformat(), "val": a, **k})
    g = {"EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
         "Revenues": {"units": {"USD": rev}}, "Assets": {"units": {"USD": assets}}}
    if has_cogs:
        g["CostOfRevenue"] = {"units": {"USD": cogs}}
    return {"facts": {"us-gaap": g}}


def generate(cfg: dict, n_stocks: int = 120, seed: int = 7, end: str = "2026-09-23",
             plant: str | None = None) -> dict:
    """Write a fake universe, price panel and PIT fundamentals under cfg's data dir.

    plant="momentum" gives each stock a persistent drift, so momentum SHOULD work.
    Used by a test to show the pipeline can find an edge that is really there.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(cfg["dates"]["data_start"], end)
    mkt = rng.normal(0.07 / 252, 0.17 / np.sqrt(252), len(dates))
    frames, uni = {}, []
    for t in cfg["universe"]["etfs"]:
        if t == cfg["universe"]["benchmark"]:
            beta, vol = 1.0, 0.01  # the fake SPY is the fake market
        elif t in ("IEF", "TLT", "GLD"):
            beta, vol = 0.0, 0.08
        else:
            beta, vol = rng.uniform(0.6, 1.2), 0.10
        frames[t] = _prices(rng, dates, 0.02 if beta == 0 else 0.0, vol, beta, mkt)
        uni.append({"ticker": t, "sector": "ETF", "source": "etf", "kind": "etf"})
    rows = []
    for k in range(n_stocks):
        t = f"S{k:03d}"
        sector = SECTORS[k % len(SECTORS)]
        mu = rng.normal(0.0, 0.25) if plant == "momentum" else 0.0
        df = _prices(rng, dates, mu, rng.uniform(0.2, 0.4), rng.uniform(0.7, 1.3), mkt)
        if k % 10 == 3:  # some names list later
            df = df[df.index >= "2012-06-01"]
        frames[t] = df
        # most names joined the index long ago; about 3 in 10 join at a random later date
        added = (pd.Timestamp("1995-01-01") if rng.random() < 0.7
                 else pd.Timestamp(rng.choice(dates[: len(dates) * 3 // 4])))
        uni.append({"ticker": t, "sector": sector, "source": "synthetic", "kind": "stock", "added": added})
        facts = _facts(rng, t, 2009, dates[-1], has_cogs=(k % 7 != 0))
        rows.append(extract_rows(facts, t, cfg))
    out = data_dir(cfg) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for f in ["open", "high", "low", "close", "volume"]:
        pd.DataFrame({t: d[f] for t, d in frames.items()}).reindex(dates).to_parquet(out / f"prices_{f}.parquet")
    pd.DataFrame(uni).to_csv(out / "tickers.csv", index=False)
    pit = build_pit(pd.concat(rows, ignore_index=True), cfg)
    pit.to_parquet(out / "fundamentals_pit.parquet")
    return {"days": len(dates), "tickers": len(frames), "fundamental_rows": len(pit)}
