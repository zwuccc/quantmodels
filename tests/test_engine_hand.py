"""A tiny backtest you can check by hand.

Two assets, six days, 1000 of cash, costs 15 bps per side.

Day   A open/close   B open/close
d1    10 / 10        20 / 20
d2    10 / 10        20 / 20      <- close d2: target A 40%, B 40%
d3    10 / 11        20 / 21      <- fill at d3 OPEN: buy 40 A @10, 20 B @20
d4    12 / 12        21 / 22      <- close d3 target A = 0, fill at d4 OPEN @12
d5    12 / 12        22 / 23
d6    12 / 12        22 / 22

With costs:
d3: spend 800, cost 800 * 0.0015 = 1.20, cash 198.80
    close equity = 198.80 + 40*11 + 20*21 = 1058.80
d4: sell 40 A @12 = 480, cost 0.72, cash = 678.08
    close equity = 678.08 + 20*22 = 1118.08
d5: 678.08 + 20*23 = 1138.08
d6: 678.08 + 20*22 = 1118.08
Trade A: invested 400.60, got back 479.28, return 19.64%, held 1 day.
"""
import numpy as np
import pandas as pd
import pytest

from qm.engine.backtest import run_backtest

D = pd.date_range("2020-01-06", periods=6, freq="B")
OPEN = pd.DataFrame({"A": [10, 10, 10, 12, 12, 12], "B": [20, 20, 20, 21, 22, 22]}, index=D, dtype=float)
CLOSE = pd.DataFrame({"A": [10, 10, 11, 12, 12, 12], "B": [20, 20, 21, 22, 23, 22]}, index=D, dtype=float)
TGT = pd.DataFrame({"A": [0.4, 0.0], "B": [0.4, np.nan]}, index=[D[1], D[2]])


def test_hand_checked_with_costs():
    res = run_backtest(OPEN, CLOSE, TGT, cost_rate=0.0015, initial_capital=1000.0)
    assert res.equity.round(2).tolist() == [1000.0, 1000.0, 1058.80, 1118.08, 1138.08, 1118.08]
    assert res.costs.round(2).tolist() == [0, 0, 1.20, 0.72, 0, 0]
    t = res.trades.set_index("ticker")
    assert t.loc["A", "invested"] == pytest.approx(400.60)
    assert t.loc["A", "proceeds"] == pytest.approx(479.28)
    assert t.loc["A", "ret"] == pytest.approx(479.28 / 400.60 - 1)
    assert t.loc["A", "days"] == 1 and not t.loc["A", "open"]
    assert t.loc["B", "open"]  # B is still held at the end
    assert res.final_weights["B"] == pytest.approx(440 / 1118.08)


def test_hand_checked_without_costs():
    res = run_backtest(OPEN, CLOSE, TGT, cost_rate=0.0, initial_capital=1000.0)
    assert res.equity.round(2).tolist() == [1000.0, 1000.0, 1060.0, 1120.0, 1140.0, 1120.0]


def test_fill_is_next_open_not_signal_close():
    # A is sold at the d4 open (12), not the d3 close (11) when the signal fired.
    res = run_backtest(OPEN, CLOSE, TGT, cost_rate=0.0, initial_capital=1000.0)
    assert res.trades.set_index("ticker").loc["A", "proceeds"] == pytest.approx(40 * 12)
