"""Universe: current S&P 500 and Nasdaq 100 members, plus the ETF list.

These are TODAY's members. Stocks that were dropped or went bust are missing,
which is survivorship bias (rule 6).

If Wikipedia can't be reached, drop a CSV at data/raw/universe/tickers_manual.csv
with columns ticker,sector and it will be used instead.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from qm.config import data_dir

UA = {"User-Agent": "Mozilla/5.0 (research script; quantmodels)"}


def _cached_html(url: str, path: Path, refresh: bool) -> str:
    if path.exists() and not refresh:
        return path.read_text()
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.text


def _find_table(html: str, symbol_cols: tuple[str, ...]) -> pd.DataFrame:
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).strip(): c for c in t.columns}
        for sc in symbol_cols:
            if sc in cols:
                sector_col = (next((cols[c] for c in cols if "GICS Sector" in c or c == "Sector"), None)
                              or next((cols[c] for c in cols if "Industry" in c), None))  # Nasdaq list uses ICB
                out = pd.DataFrame({"ticker": t[cols[sc]].astype(str).str.strip()})
                out["sector"] = t[sector_col].astype(str) if sector_col is not None else ""
                added = next((cols[c] for c in cols if c.startswith("Date added")), None)
                out["added"] = pd.to_datetime(t[added], errors="coerce") if added is not None else pd.NaT
                return out
    raise ValueError("no constituent table found")


def normalize(ticker: str) -> str:
    """Wikipedia uses BRK.B; we store BRK-B (the yfinance form)."""
    return ticker.strip().upper().replace(".", "-")


def build_universe(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    raw = data_dir(cfg) / "raw" / "universe"
    out_path = data_dir(cfg) / "processed" / "tickers.csv"
    manual = raw / "tickers_manual.csv"
    if out_path.exists() and not refresh:
        return pd.read_csv(out_path)

    if manual.exists():
        stocks = pd.read_csv(manual)
        stocks["source"] = "manual"
    else:
        sp = _find_table(_cached_html(cfg["universe"]["sp500_url"], raw / "sp500.html", refresh), ("Symbol",))
        sp["source"] = "sp500"
        nd = _find_table(_cached_html(cfg["universe"]["ndx_url"], raw / "ndx.html", refresh), ("Ticker", "Symbol"))
        nd["source"] = "ndx"
        stocks = pd.concat([sp, nd], ignore_index=True)

    stocks["ticker"] = stocks["ticker"].map(normalize)
    stocks = (stocks.groupby("ticker", as_index=False)
              .agg(sector=("sector", "first"), added=("added", "first"),
                   source=("source", lambda s: "+".join(sorted(set(s))))))
    stocks["kind"] = "stock"
    etfs = pd.DataFrame({"ticker": cfg["universe"]["etfs"], "sector": "ETF", "source": "etf", "kind": "etf"})
    stocks = stocks[~stocks["ticker"].isin(etfs["ticker"])]
    uni = pd.concat([etfs, stocks], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(out_path, index=False)
    return uni
