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
    Point in time: each event only uses surprises that were public by its own
    date. Companies sometimes report an old quarter for the first time in a
    later filing; that quarter must not change an earlier event. Where two
    rows could serve, the earliest filed one is used, so loading later
    filings never changes an earlier result.
    """
    t = fund.table
    q = t[(t["concept"] == "eps") & (t["period"] == "Q")].sort_values(["ticker", "end", "filed"], kind="stable")
    day = np.timedelta64(1, "D")
    out = []
    for tk, g in q.groupby("ticker"):
        g = g.drop_duplicates("end", keep="first")  # earliest filed value for each quarter
        ends = g["end"].to_numpy("datetime64[ns]")
        vals, filed = g["value"].to_numpy(), g["filed"].to_numpy("datetime64[ns]")
        sur_end, sur_avail, sur_val = [], [], []
        for i in range(len(g)):
            target = ends[i] - 365 * day
            j = np.where((np.abs((ends - target) / day) <= 20) & (ends < ends[i]))[0]
            if len(j) == 0:
                continue
            j = j[np.argmin(filed[j])]  # earliest filed match
            sur_end.append(ends[i])
            sur_avail.append(max(filed[i], filed[j]))
            sur_val.append(vals[i] - vals[j])
        sur_end, sur_avail, sur_val = np.array(sur_end), np.array(sur_avail), np.array(sur_val)
        for k in range(len(sur_end)):
            known = np.where((sur_end < sur_end[k]) & (sur_avail <= sur_avail[k]))[0]
            if len(known) < n_surprises:
                continue
            prev = known[np.argsort(sur_end[known], kind="stable")][-n_surprises:]
            if (sur_end[k] - sur_end[prev[0]]) / day > (n_surprises + 1) * 95:
                continue  # gap in the history, not n consecutive quarters
            sd = np.std(sur_val[prev], ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                continue
            sd = max(sd, min_sd)
            out.append({"ticker": tk, "end": pd.Timestamp(sur_end[k]), "filed": pd.Timestamp(sur_avail[k]),
                        "sue": sur_val[k] / sd})
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
    # Two ways to get gross profit, each dated by its own filings. Both are kept:
    # if the GrossProfit tag is first filed later than revenue and cost, the
    # earlier revenue minus cost version was what a trader could see at first.
    rc = rev.merge(cogs, on=key, how="inner")
    rc["gross"] = rc["rev"] - rc["cogs"]
    rc["f_gross"] = rc[["f_rev", "f_cogs"]].max(axis=1)
    gp = gp.rename(columns={"gp": "gross", "f_gp": "f_gross"})
    m = pd.concat([rc[key + ["gross", "f_gross"]], gp[key + ["gross", "f_gross"]]], ignore_index=True)
    a = inst[["ticker", "end", "value", "filed"]].rename(columns={"value": "assets", "filed": "f_assets"})
    m = m.merge(a, on=["ticker", "end"], how="inner")
    m = m[m["assets"] > 0]
    m["filed"] = m[["f_gross", "f_assets"]].max(axis=1)
    m["gpa"] = m["gross"] / m["assets"]
    if bounds is not None:
        m = m[m["gpa"].between(*bounds)]
    m = m.sort_values(["ticker", "filed", "end"], kind="stable")
    # at most one row per company per filing date: the newest period, GrossProfit tag preferred
    m = m.drop_duplicates(["ticker", "filed"], keep="last")
    return m[["ticker", "end", "filed", "gpa"]].reset_index(drop=True)
