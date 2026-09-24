import numpy as np
import pandas as pd
import pytest

from qm.engine.backtest import buy_and_hold_targets, run_backtest
from tests.helpers import random_market


def test_signal_on_day_t_never_changes_day_t_return():
    md = random_market()
    d = md.calendar
    t1 = pd.DataFrame({"AAA": [1.0]}, index=[d[100]])
    t2 = pd.DataFrame({"AAA": [1.0, 0.0]}, index=[d[100], d[200]])
    a = run_backtest(md.open, md.close, t1, 0.0015)
    b = run_backtest(md.open, md.close, t2, 0.0015)
    # identical through the close of day 200, even though b's exit was decided then
    pd.testing.assert_series_equal(a.equity.iloc[:201], b.equity.iloc[:201])
    assert a.equity.iloc[201] != b.equity.iloc[201]


def test_costs_equal_traded_value_times_rate():
    md = random_market()
    d = md.calendar
    tgt = pd.DataFrame({"AAA": [0.5, 0.0], "BBB": [0.5, 0.0]}, index=[d[10], d[50]])
    res = run_backtest(md.open, md.close, tgt, 0.0015, initial_capital=1.0)
    # rebuild traded money from trades: buys + sells
    buys = res.trades["invested"].sum() / 1.0015
    sells = (res.trades["proceeds"] / (1 - 0.0015)).sum()
    assert res.costs.sum() == pytest.approx((buys + sells) * 0.0015, rel=1e-9)
    free = run_backtest(md.open, md.close, tgt, 0.0)
    assert free.costs.sum() == 0
    assert free.equity.iloc[-1] > res.equity.iloc[-1]


def test_nan_leaves_position_alone_and_zero_exits():
    md = random_market()
    d = md.calendar
    tgt = pd.DataFrame({"AAA": [0.3, np.nan, 0.0], "BBB": [0.3, 0.0, np.nan]}, index=[d[10], d[20], d[30]])
    res = run_backtest(md.open, md.close, tgt, 0.0015)
    t = res.trades.set_index("ticker")
    assert t.loc["BBB", "exit_date"] == d[21]
    assert t.loc["AAA", "exit_date"] == d[31]
    assert len(res.trades) == 2


def test_no_leverage_when_targets_exceed_cash():
    md = random_market()
    tgt = pd.DataFrame({"AAA": [0.8], "BBB": [0.8]}, index=[md.calendar[5]])
    res = run_backtest(md.open, md.close, tgt, 0.0015)
    assert res.exposure.max() <= 1.0 + 1e-9


def test_missing_open_skips_trade_and_logs():
    md = random_market()
    o = md.open.copy()
    o.iloc[11, o.columns.get_loc("AAA")] = np.nan
    tgt = pd.DataFrame({"AAA": [0.5]}, index=[md.calendar[10]])
    res = run_backtest(o, md.close, tgt, 0.0015)
    assert len(res.trades) == 0 and res.skipped and "no open price" in res.skipped[0][2]


def test_buy_and_hold_benchmark_is_fully_invested_after_costs():
    md = random_market()
    tgt = buy_and_hold_targets(md.close, "SPY", md.calendar[0])
    res = run_backtest(md.open, md.close, tgt, 0.0015)
    growth = md.close["SPY"].iloc[-1] / md.open["SPY"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(growth / 1.0015, rel=1e-9)
