import numpy as np
import pandas as pd
import pytest

from qm.engine.metrics import cagr, max_drawdown, sharpe
from qm.engine.stats import deflated_sharpe, expected_max_sr, psr


def test_dsr_matches_paper_example():
    # Bailey & Lopez de Prado (2014) worked example: annual SR 2.5 over 1250 days,
    # 100 trials, variance of annual SR 0.5, skew -3, kurtosis 10 -> DSR about 0.90
    sr = 2.5 / np.sqrt(250)
    dsr = deflated_sharpe(sr, T=1250, skew=-3, kurt=10, var_sr=0.5 / 250, n_trials=100)
    assert dsr == pytest.approx(0.90, abs=0.01)


def test_dsr_with_one_trial_is_psr_against_zero():
    assert deflated_sharpe(0.05, 1000, 0, 3, 0.001, 1) == pytest.approx(psr(0.05, 0.0, 1000, 0, 3))


def test_dsr_falls_as_trials_rise():
    vals = [deflated_sharpe(0.05, 1000, 0, 3, 0.0004, n) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert expected_max_sr(0.0004, 100) > expected_max_sr(0.0004, 10)


def test_metrics_on_known_series():
    d = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.50, 0.20, 0.0], index=d)
    assert max_drawdown(r) == pytest.approx(-0.5)  # peak 1.10, trough 0.55
    flat = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    assert cagr(flat) == pytest.approx(1.001 ** 252 - 1)
    assert sharpe(flat) == 0.0  # zero variance
