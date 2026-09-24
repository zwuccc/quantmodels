"""A strategy that peeks at tomorrow on purpose. The look ahead check must catch it."""
import pandas as pd
import pytest

from qm.engine.backtest import run_backtest
from qm.engine.leakcheck import LookAheadError, check_no_lookahead
from qm.engine.metrics import summarize
from qm.engine.sanity import too_good_flags
from tests.helpers import random_market


def cheating(md):
    """Hold AAA only on days when TOMORROW's close is higher. Uses the future."""
    up = md.close["AAA"].shift(-1) > md.close["AAA"]
    return pd.DataFrame({"AAA": up.astype(float)})


def honest(md):
    """Hold AAA when today's close is above the 20 day average. Past only."""
    c = md.close["AAA"]
    return pd.DataFrame({"AAA": (c > c.rolling(20).mean()).astype(float)})


def test_leaky_strategy_is_caught():
    md = random_market()
    with pytest.raises(LookAheadError):
        check_no_lookahead(cheating, md)


def test_honest_strategy_passes():
    check_no_lookahead(honest, random_market())


def test_peeking_at_the_signal_close_makes_results_too_good():
    # Rule 9 in action: a cheat that trades on the same day it sees the answer
    # looks amazing, and the sanity flags fire.
    md = random_market(n_days=1500)
    up = md.close["AAA"] > md.open["AAA"]            # today's move, known only at today's close
    tgt = pd.DataFrame({"AAA": up.shift(-1).astype(float)})   # placed a day early on purpose
    res = run_backtest(md.open, md.close, tgt.dropna(), cost_rate=0.0)
    # Filled at the open of the day whose close it already knew: pure foresight.
    m = summarize(res)
    assert m["sharpe"] > 3
    cfg = {"sanity": {"max_sharpe": 2.0, "max_win_rate": 0.7, "max_smoothness_r2": 0.98, "min_maxdd_long": 0.05}}
    assert too_good_flags(m, cfg)
