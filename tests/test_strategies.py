import numpy as np
import pandas as pd
import pytest

from qm.config import load_config
from qm.data.fundamentals import Fundamentals, sue_events
from qm.engine.leakcheck import check_no_lookahead
from qm.research.common import ORDER, load_market, params_for, run_window
from qm.strategies import build_targets
from qm.strategies.indicators import month_start, prior_high, rsi
from qm.synthetic import generate
from tests.helpers import random_market


def test_rsi2_hand_values():
    c = pd.Series([10, 11, 10, 12, 11], dtype=float)
    r = rsi(c, 2)
    assert np.isnan(r.iloc[1])
    assert r.iloc[2] == pytest.approx(50.0)
    assert r.iloc[3] == pytest.approx(100 - 100 / (1 + 1.25 / 0.25))
    assert r.iloc[4] == pytest.approx(50.0)
    assert rsi(pd.Series(np.arange(1.0, 20)), 2).iloc[-1] == 100.0
    assert rsi(pd.Series(np.arange(20.0, 1, -1)), 2).iloc[-1] == pytest.approx(0.0)


def test_prior_high_excludes_today():
    h = pd.Series([1, 2, 3, 10, 4], dtype=float)
    ph = prior_high(h, 3)
    assert ph.iloc[3] == 3.0   # today's 10 is not part of its own breakout level
    assert ph.iloc[4] == 10.0


def test_month_start_only_uses_past():
    d = pd.bdate_range("2020-01-27", "2020-03-04")
    ms = month_start(d)
    assert list(d[ms]) == [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")]
    assert (month_start(d[:10]) == ms[:10]).all()


def test_trend_holds_asset_above_average_and_ief_below():
    md = random_market(n_days=400)
    c = md.close.copy()
    c["SPY"] = np.r_[np.linspace(100, 150, 300), np.linspace(150, 100, 100)]
    md.close = c
    p = {"kind": "trend_sma", "asset": "SPY", "safe": "IEF", "sma": 50, "check": "daily"}
    t = build_targets(md, p)
    assert t.iloc[0].to_dict() == {"SPY": 1.0, "IEF": 0.0}
    assert t.iloc[-1].to_dict() == {"SPY": 0.0, "IEF": 1.0}
    assert len(t) == 2  # only change days are sent


def test_meanrev_entry_and_time_exit():
    md = random_market(n_days=300, tickers=("SPY", "IEF", "AAA"))
    c = md.close.copy()
    base = np.linspace(50, 100, 300)
    base[250:253] = [90, 85, 80]           # sharp dip well above the 200 day average
    base[253:] = 79                          # then flat below the 5 day average... never recovers
    c["AAA"] = base
    md.close = c
    p = {"kind": "meanrev", "rsi_len": 2, "rsi_entry": 10, "trend_sma": 200, "exit_sma": 5, "max_hold": 10, "slots": 10}
    t = build_targets(md, p)["AAA"].dropna()
    entry = t[t > 0].index[0]
    assert entry == md.calendar[250]  # the drop from ~99.8 to 90 already sends RSI(2) to ~1.7
    exit_ = t[(t == 0) & (t.index > entry)].index[0]
    held = md.calendar.get_loc(exit_) - md.calendar.get_loc(entry)
    assert held == 10  # price never closes above its 5 day average, so the time exit fires


def test_breakout_sizes_by_atr():
    md = random_market(n_days=300, tickers=("SPY", "IEF", "EEM"))
    p = {"kind": "breakout", "universe": "etf", "entry": 55, "exit": 20, "atr": 20, "risk_per_trade": 0.01, "max_weight": 0.2}
    t = build_targets(md, p)
    w = t.stack().loc[lambda s: s > 0]
    assert len(w) > 0 and (w <= 0.2 + 1e-12).all()


def test_momentum_picks_the_strongest_12_minus_1():
    d = pd.bdate_range("2010-01-04", periods=320)
    names = [f"S{i:02d}" for i in range(30)]
    drift = np.linspace(-0.001, 0.002, 30)
    close = pd.DataFrame(100 * np.exp(np.outer(np.arange(320), drift)), index=d, columns=names)
    md = random_market(n_days=320)
    md.close = close
    md.universe = pd.DataFrame({"ticker": names, "sector": "Tech", "kind": "stock"})
    t = build_targets(md, {"kind": "momentum", "lookback": 252, "skip": 21, "top_frac": 0.10})
    last = t.iloc[-1]
    assert set(last[last > 0].index) == {"S29", "S28", "S27"}
    assert last.sum() == pytest.approx(1.0)


def test_sue_value_by_hand():
    rows = []
    vals = [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.0, 1.3, 1.0, 1.1, 1.3, 0.9, 1.5]
    ends = pd.date_range("2012-03-31", periods=len(vals), freq="QE")
    for e, v in zip(ends, vals):
        rows.append({"ticker": "X", "concept": "eps", "tag": "t", "start": e - pd.Timedelta(days=89), "end": e, "days": 89,
                     "period": "Q", "value": v, "form": "10-Q", "accn": "", "filed": e + pd.Timedelta(days=35), "derived": False})
    ev = sue_events(Fundamentals(pd.DataFrame(rows)), 8)
    surprises = [vals[i] - vals[i - 4] for i in range(4, len(vals))]  # 9 surprises
    expected = surprises[8] / np.std(surprises[:8], ddof=1)
    assert len(ev) == 1 and ev["sue"].iloc[0] == pytest.approx(expected)


@pytest.fixture(scope="module")
def small_market(tmp_path_factory):
    cfg = load_config(synthetic=True)
    d = tmp_path_factory.mktemp("syn")
    cfg["paths"] = {"data_dir": d / "data", "results_dir": d / "results"}
    generate(cfg, n_stocks=60, seed=11, end="2016-06-30")
    return cfg, load_market(cfg)


@pytest.mark.parametrize("name", ORDER)
def test_every_strategy_passes_the_look_ahead_check(small_market, name):
    cfg, md = small_market
    p = params_for(cfg, name)
    t = build_targets(md, p)
    assert len(t.dropna(how="all")) > 0, "strategy produced no signals"
    check_no_lookahead(lambda m: build_targets(m, p), md, n_cuts=2, seed=3)


def test_planted_momentum_is_found(tmp_path):
    # When stocks really do have persistent drift, momentum must beat equal weight.
    cfg = load_config(synthetic=True)
    cfg["paths"] = {"data_dir": tmp_path / "data", "results_dir": tmp_path / "results"}
    generate(cfg, n_stocks=80, seed=5, end="2015-12-31", plant="momentum")
    md = load_market(cfg)
    run = run_window(md, cfg, build_targets(md, params_for(cfg, "D")), "2006-01-01", "2015-12-31", ew_baseline=True)
    assert run["m_on"]["cagr"] > run["m_ew"]["cagr"] + 0.05


def test_dated_universe_never_picks_a_stock_before_it_joined():
    d = pd.bdate_range("2010-01-04", periods=400)
    names = [f"S{i:02d}" for i in range(30)]
    drift = np.linspace(-0.001, 0.002, 30)
    md = random_market(n_days=400)
    md.close = pd.DataFrame(100 * np.exp(np.outer(np.arange(400), drift)), index=d, columns=names)
    joined = pd.Timestamp("2011-03-01")  # the three strongest names join late
    md.universe = pd.DataFrame({"ticker": names, "sector": "Tech", "kind": "stock",
                                "added": [joined if n in ("S27", "S28", "S29") else pd.Timestamp("2000-01-01") for n in names]})
    p = {"kind": "momentum", "lookback": 252, "skip": 21, "top_frac": 0.10, "min_names": 20}
    plain = build_targets(md, p)
    dated = build_targets(md, {**p, "dated_universe": True})
    early = dated[dated.index < joined]
    assert (early[["S27", "S28", "S29"]] == 0).all().all()
    assert (plain.loc[plain.index < joined, "S29"] > 0).any()     # the plain version did pick it
    assert (dated.loc[dated.index >= joined, "S29"] > 0).any()    # and the dated one does after it joins
