"""Point in time fundamentals from SEC XBRL companyfacts.

Rules:
- Every value keeps its SEC filed date. A value is usable from its filed date
  on, never from its period end date.
- For each (company, tag, period) we keep the FIRST filed value. Restatements
  filed later are ignored, because that's what a trader saw at the time.
- Q4 has no 10-Q, so Q4 EPS is derived as annual minus the three quarters,
  dated by the latest of those filings. Derived rows are flagged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qm.config import data_dir

UNITS = {"eps": "USD/shares"}  # everything else is USD
PIT_COLS = ["ticker", "concept", "tag", "start", "end", "days", "period",
            "value", "form", "accn", "filed", "derived"]


def extract_rows(facts: dict, ticker: str, cfg: dict) -> pd.DataFrame:
    """Flatten one companyfacts JSON into rows for the tags we care about."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    forms = set(cfg["fundamentals"]["forms"])
    rows = []
    for concept, tags in cfg["fundamentals"]["tags"].items():
        unit = UNITS.get(concept, "USD")
        for prio, tag in enumerate(tags):
            for r in gaap.get(tag, {}).get("units", {}).get(unit, []):
                if r.get("form") not in forms or r.get("val") is None or not r.get("filed"):
                    continue
                rows.append({
                    "ticker": ticker, "concept": concept, "tag": tag, "prio": prio,
                    "start": r.get("start"), "end": r["end"], "value": float(r["val"]),
                    "form": r["form"], "accn": r.get("accn", ""), "filed": r["filed"],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def build_pit(rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """First filed value per period, best tag per period, periods classified."""
    if rows.empty:
        return pd.DataFrame(columns=PIT_COLS)
    rows = rows[~(rows["filed"] < rows["end"])]  # filed before its own period ended: a typo in the filing
    df = rows.sort_values(["filed", "accn"])
    key = ["ticker", "concept", "tag", "start", "end"]
    df = df.drop_duplicates(subset=key, keep="first")  # first filed wins
    df = df.sort_values(["prio", "filed"])
    df = df.drop_duplicates(subset=["ticker", "concept", "start", "end"], keep="first")  # best tag wins
    df["days"] = (df["end"] - df["start"]).dt.days
    qd, ad = cfg["fundamentals"]["quarter_days"], cfg["fundamentals"]["annual_days"]
    df["period"] = np.select(
        [df["start"].isna(), df["days"].between(*qd), df["days"].between(*ad)],
        ["I", "Q", "A"], default="other")
    df = df[df["period"] != "other"].copy()
    df["derived"] = False
    df = pd.concat([df, _derive_q4_eps(df)], ignore_index=True)
    return df[PIT_COLS].sort_values(["ticker", "concept", "end", "filed"]).reset_index(drop=True)


def _derive_q4_eps(df: pd.DataFrame) -> pd.DataFrame:
    eps = df[df["concept"] == "eps"]
    out = []
    for t, g in eps.groupby("ticker"):
        q = g[g["period"] == "Q"]
        for _, a in g[g["period"] == "A"].iterrows():
            inside = q[(q["start"] >= a["start"] - pd.Timedelta(days=5)) & (q["end"] <= a["end"] + pd.Timedelta(days=5))]
            if (inside["end"] >= a["end"] - pd.Timedelta(days=20)).any():
                continue  # a real Q4 value exists
            if len(inside) != 3:
                continue
            last_q_end = inside["end"].max()
            out.append({
                "ticker": t, "concept": "eps", "tag": a["tag"] + ":derivedQ4",
                "start": last_q_end + pd.Timedelta(days=1), "end": a["end"],
                "days": (a["end"] - last_q_end).days, "period": "Q",
                "value": a["value"] - inside["value"].sum(), "form": a["form"], "accn": a["accn"],
                "filed": max(a["filed"], inside["filed"].max()), "derived": True,
            })
    return pd.DataFrame(out, columns=PIT_COLS)


class Fundamentals:
    """The only way strategies read fundamentals."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.sort_values("filed").reset_index(drop=True)

    def as_of(self, date) -> pd.DataFrame:
        """Rows filed on or before date. A value filed on X is invisible on X - 1."""
        return self.table[self.table["filed"] <= pd.Timestamp(date)]

    def clip(self, date) -> "Fundamentals":
        return Fundamentals(self.as_of(date))

    @property
    def empty(self) -> bool:
        return self.table.empty


def pit_path(cfg: dict) -> Path:
    return data_dir(cfg) / "processed" / "fundamentals_pit.parquet"


def load_fundamentals(cfg: dict) -> Fundamentals:
    p = pit_path(cfg)
    if not p.exists():
        return Fundamentals(pd.DataFrame(columns=PIT_COLS))
    t = pd.read_parquet(p)
    for c in ("start", "end", "filed"):  # one date precision everywhere, or merges refuse to match
        t[c] = t[c].astype("datetime64[ns]")
    return Fundamentals(t)


# ---- signal inputs built from the PIT table -------------------------------

def sue_events(fund: Fundamentals, n_surprises: int, min_sd: float = 0.0) -> pd.DataFrame:
    """Standardized unexpected earnings, one row per quarterly EPS filing.

    surprise_q = EPS_q - EPS_(same quarter last year)
    SUE_q = surprise_q / std(previous n_surprises surprises)   (current one excluded)
    The event date is when both EPS values were public (the later filed date).
    min_sd floors the standard deviation: EPS is reported to the cent, so a
    spread below a cent is rounding noise, not a real baseline.
    """
    t = fund.table
    q = t[(t["concept"] == "eps") & (t["period"] == "Q")].sort_values(["ticker", "end"])
    out = []
    for tk, g in q.groupby("ticker"):
        g = g.drop_duplicates("end", keep="first")
        ends = g["end"].to_numpy()
        vals, filed = g["value"].to_numpy(), g["filed"].to_numpy()
        surprises = []  # (end, avail, surprise)
        for i in range(len(g)):
            target = ends[i] - np.timedelta64(365, "D")
            j = np.where(np.abs((ends - target) / np.timedelta64(1, "D")) <= 20)[0]
            j = j[j < i]
            if len(j) == 0:
                continue
            j = j[-1]
            surprises.append((ends[i], max(filed[i], filed[j]), vals[i] - vals[j]))
        for k in range(n_surprises, len(surprises)):
            end, avail, s = surprises[k]
            prev = surprises[k - n_surprises:k]
            span = (end - prev[0][0]) / np.timedelta64(1, "D")
            if span > (n_surprises + 1) * 95:
                continue  # gap in the history, not n consecutive quarters
            if any(p[1] > avail for p in prev):
                continue
            sd = np.std([p[2] for p in prev], ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                continue
            sd = max(sd, min_sd)
            out.append({"ticker": tk, "end": pd.Timestamp(end), "filed": pd.Timestamp(avail), "sue": s / sd})
    return pd.DataFrame(out, columns=["ticker", "end", "filed", "sue"])


def annual_gross_profitability(fund: Fundamentals, bounds: tuple[float, float] | None = None) -> pd.DataFrame:
    """(Revenue - cost of revenue) / total assets, one row per annual filing.

    Uses the GrossProfit tag when a company reports it, else revenue minus
    cost of revenue from the same annual period. Missing pieces mean no row.
    Values outside bounds are unit or tagging errors in the filing and are dropped.
    """
    t = fund.table
    ann = t[t["period"] == "A"]
    inst = t[(t["concept"] == "assets") & (t["period"] == "I")]
    key = ["ticker", "start", "end"]
    rev = ann[ann["concept"] == "revenue"][key + ["value", "filed"]].rename(columns={"value": "rev", "filed": "f_rev"})
    cogs = ann[ann["concept"] == "cogs"][key + ["value", "filed"]].rename(columns={"value": "cogs", "filed": "f_cogs"})
    gp = ann[ann["concept"] == "gross_profit"][key + ["value", "filed"]].rename(columns={"value": "gp", "filed": "f_gp"})
    m = rev.merge(cogs, on=key, how="outer").merge(gp, on=key, how="outer")
    m["gross"] = m["gp"].where(m["gp"].notna(), m["rev"] - m["cogs"])
    m["f_gross"] = m["f_gp"].where(m["gp"].notna(), m[["f_rev", "f_cogs"]].max(axis=1))
    m = m.dropna(subset=["gross"])
    a = inst[["ticker", "end", "value", "filed"]].rename(columns={"value": "assets", "filed": "f_assets"})
    m = m.merge(a, on=["ticker", "end"], how="inner")
    m = m[m["assets"] > 0]
    m["filed"] = m[["f_gross", "f_assets"]].max(axis=1)
    m["gpa"] = m["gross"] / m["assets"]
    if bounds is not None:
        m = m[m["gpa"].between(*bounds)]
    return m[["ticker", "end", "filed", "gpa"]].sort_values(["ticker", "filed"]).reset_index(drop=True)
