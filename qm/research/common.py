"""Shared plumbing for the research phases."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qm.config import cost_rate, data_dir, results_dir, strategy_params
from qm.data.fundamentals import load_fundamentals
from qm.data.market import MarketData
from qm.engine.backtest import buy_and_hold_targets, run_backtest
from qm.engine.metrics import summarize
from qm.engine.sanity import too_good_flags
from qm.engine.splits import Splits, make_splits
from qm.engine.stats import deflated_sharpe
from qm.engine.trials import TrialLog
from qm.strategies import build_targets, uses_single_stocks

SURVIVORSHIP_WARNING = (
    "SURVIVORSHIP WARNING: the stock universe is today's S&P 500 and Nasdaq 100 members. "
    "Companies that were dropped, merged away or went bust are missing, so single stock "
    "results look better than reality. Treat them as an upper bound.")

SYNTHETIC_WARNING = (
    "SYNTHETIC DATA: these numbers come from random fake prices, not real markets. "
    "They only show that the pipeline runs. They say nothing about any strategy.")

ORDER = ["A1", "A2", "A3", "B", "C", "C_stocks", "D", "E", "F"]


# ---- loading ---------------------------------------------------------------

def panel_path(cfg: dict, field: str) -> Path:
    return data_dir(cfg) / "processed" / f"prices_{field}.parquet"


def load_market(cfg: dict) -> MarketData:
    fields = ["open", "high", "low", "close", "volume"]
    missing = [f for f in fields if not panel_path(cfg, f).exists()]
    if missing:
        raise FileNotFoundError("price panel not built yet. Run: python -m qm panel "
                                "(after universe and prices), or python -m qm synthetic")
    px = {f: pd.read_parquet(panel_path(cfg, f)) for f in fields}
    uni = pd.read_csv(data_dir(cfg) / "processed" / "tickers.csv")
    return MarketData(**px, fundamentals=load_fundamentals(cfg), universe=uni)


def research_view(md: MarketData, cfg: dict, holdout: bool = False) -> tuple[MarketData, Splits]:
    """The data a research step may see. Without holdout=True it ends before the holdout."""
    splits = make_splits(cfg, md.calendar)
    view = md.clip(splits.snapshot_end) if holdout else md.clip(splits.dev_end)
    return view, splits


def trial_log(cfg: dict) -> TrialLog:
    return TrialLog(results_dir(cfg) / "trials.csv")


def params_for(cfg: dict, name: str, overrides: dict | None = None) -> dict:
    p = strategy_params(cfg, name, overrides)
    p["financial_sectors"] = cfg["universe"]["financial_sectors"]
    return p


def tunable(p: dict) -> dict:
    """The parameters that define a trial (drops bookkeeping keys)."""
    return {k: v for k, v in p.items() if k not in ("financial_sectors",)}


def grid_combos(cfg: dict, name: str) -> list[dict]:
    g = cfg["grids"].get(name, {})
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[g[k] for k in keys])] or [{}]


# ---- running ---------------------------------------------------------------

def seeded(targets: pd.DataFrame, start: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Targets from start on. If the last row before start was a full portfolio
    (no NaN, as in A, D, F), restate it on the first day so the run doesn't sit
    in cash waiting for the next change. B, C, E start flat."""
    before = targets[targets.index < start]
    after = targets[targets.index >= start]
    first = calendar[calendar >= start]
    if len(before) and len(first) and not before.iloc[-1].isna().any():
        if len(after) == 0 or after.index[0] != first[0]:
            seed = before.iloc[[-1]].copy()
            seed.index = [first[0]]
            after = pd.concat([seed, after])
    return after


def equal_weight_targets(md: MarketData) -> pd.DataFrame:
    """Every stock in the universe, equal weight, rebalanced on the first trading
    day of each month. A second baseline for stock strategies: it shows how much
    return comes from just owning today's survivors, before any signal."""
    from qm.strategies.indicators import month_start
    names = md.tickers("stocks")
    C = md.close[names]
    rows = C[month_start(C.index)]
    valid = rows.notna()
    return valid.div(valid.sum(axis=1), axis=0).where(valid, 0.0)


def run_window(md: MarketData, cfg: dict, targets: pd.DataFrame, start, end,
               ew_baseline: bool = False) -> dict:
    """Strategy (costs on and off) and SPY buy and hold over the same dates.
    With ew_baseline, also the equal weight stock universe over the same dates."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cal = md.calendar[(md.calendar >= start) & (md.calendar <= end)]
    tg = seeded(targets[targets.index <= end], start, md.calendar)
    if tg.empty or len(cal) < 20:
        return {}
    active = max(cal[0], tg.index[0])
    tg = tg[tg.index >= active]
    ec = cfg["engine"]
    kw = dict(initial_capital=ec["initial_capital"], cash_rate=ec["cash_rate"], start=active, end=end)
    on = run_backtest(md.open, md.close, tg, cost_rate(cfg), **kw)
    off = run_backtest(md.open, md.close, tg, 0.0, **kw)
    bench_t = buy_and_hold_targets(md.close, cfg["universe"]["benchmark"], active)
    bench = run_backtest(md.open, md.close, bench_t, cost_rate(cfg), **kw)
    rf = ec["risk_free"]
    out = {"on": on, "off": off, "bench": bench,
           "m_on": summarize(on, rf), "m_off": summarize(off, rf), "m_bench": summarize(bench, rf)}
    if ew_baseline:
        ew_t = seeded(equal_weight_targets(md.clip(end)), active, md.calendar)
        out["ew"] = run_backtest(md.open, md.close, ew_t[ew_t.index >= active], cost_rate(cfg), **kw)
        out["m_ew"] = summarize(out["ew"], rf)
    return out


def log_run(log: TrialLog, phase: str, name: str, params: dict, period: str, run: dict) -> None:
    log.log(phase, name, tunable(params), period, True, run["m_on"])
    log.log(phase, name, tunable(params), period, False, run["m_off"])


def dsr_for(log: TrialLog, name: str, m: dict) -> dict:
    """Deflated Sharpe two ways: trials of this strategy, and trials of everything."""
    out = {}
    for label, key in (("dsr_own", name), ("dsr_all", None)):
        n = log.n_trials(key)
        v = log.sharpe_variance(key)
        out[label] = deflated_sharpe(m["sharpe_daily"], m["days"], m["skew"], m["kurt"], v, n)
        out[f"n_{label[4:]}"] = n
    return out


def row_from(name: str, run: dict, cfg: dict, extra: dict | None = None) -> dict:
    m, mo, mb = run["m_on"], run["m_off"], run["m_bench"]
    r = {"strategy": name, "start": m["start"], "end": m["end"],
         "cagr": m["cagr"], "cagr_no_costs": mo["cagr"], "sharpe": m["sharpe"],
         "sharpe_no_costs": mo["sharpe"], "max_dd": m["max_dd"], "worst_year": m["worst_year"],
         "turnover": m["turnover"], "win_rate": m["win_rate"], "avg_hold_days": m["avg_hold_days"],
         "trades": m["trades"], "exposure": m["exposure"],
         "total_return": m["total_return"], "total_return_no_costs": mo["total_return"],
         "spy_cagr": mb["cagr"], "spy_sharpe": mb["sharpe"], "spy_max_dd": mb["max_dd"],
         "gap_cagr_vs_spy": m["cagr"] - mb["cagr"], "gap_sharpe_vs_spy": m["sharpe"] - mb["sharpe"],
         "flags": "; ".join(too_good_flags(m, cfg))}
    if "m_ew" in run:
        r["ew_universe_cagr"] = run["m_ew"]["cagr"]
        r["gap_cagr_vs_ew"] = m["cagr"] - run["m_ew"]["cagr"]
    if extra:
        r.update(extra)
    return r


def stitch(parts: list[pd.Series]) -> pd.Series:
    s = pd.concat([p for p in parts if p is not None and len(p)])
    return s[~s.index.duplicated(keep="first")].sort_index()


def fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    pct = {"cagr", "cagr_no_costs", "max_dd", "worst_year", "win_rate", "spy_cagr",
           "gap_cagr_vs_spy", "exposure", "spy_max_dd", "total_return", "ew_universe_cagr", "gap_cagr_vs_ew"}
    out = df[[c for c in cols if c in df.columns]].copy()
    cols = list(out.columns)
    for c in cols:
        if c in pct:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}" if "gap" in c else f"{x:.1%}")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return out.to_string(index=False)


def banner(cfg: dict, names: list[str]) -> list[str]:
    lines = []
    if cfg.get("synthetic"):
        lines.append(SYNTHETIC_WARNING)
    if any(uses_single_stocks(params_for(cfg, n)) for n in names):
        lines.append(SURVIVORSHIP_WARNING)
    return lines


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda x: None if isinstance(x, float) and np.isnan(x) else str(x)))
