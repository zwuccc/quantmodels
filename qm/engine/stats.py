"""Probabilistic and deflated Sharpe ratio.

Bailey, D. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".

All Sharpe ratios here are per period (daily), not annualized.
kurt is plain kurtosis (a normal distribution has 3).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def psr(sr: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    """Probability that the true Sharpe is above sr0."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if T < 2 or denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom)))


def expected_max_sr(var_sr: float, n_trials: int) -> float:
    """Expected best Sharpe out of n_trials tries when the true Sharpe of all is 0."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    n = float(n_trials)
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                    + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float, var_sr: float, n_trials: int) -> float:
    """PSR against the Sharpe you'd expect from the best of n_trials lucky tries."""
    return psr(sr, expected_max_sr(var_sr, n_trials), T, skew, kurt)
