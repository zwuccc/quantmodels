import pandas as pd
import pytest

from qm.engine.splits import HoldoutLocked, claim_holdout, make_splits
from qm.engine.trials import TrialLog
from tests.helpers import random_market


def test_splits_and_walk_forward(cfg):
    cal = pd.bdate_range("2005-01-03", "2026-09-23")
    s = make_splits(cfg, cal)
    assert s.holdout_start == pd.Timestamp("2024-09-24")
    w = s.wf_windows()
    assert w[0]["test_start"] == pd.Timestamp("2016-01-01") and w[-1]["label"] == "2024"
    assert w[-1]["test_end"] < s.holdout_start
    assert all(x["train_end"] < x["test_start"] for x in w)


def test_holdout_can_only_be_claimed_once(cfg):
    claim_holdout(cfg, {"x": 1})
    with pytest.raises(HoldoutLocked):
        claim_holdout(cfg, {"x": 1})


def test_research_data_never_contains_holdout(cfg):
    from qm.research.common import research_view

    md = random_market(n_days=900)
    view, splits = research_view(md, cfg, holdout=False)
    assert view.calendar[-1] < splits.holdout_start
    full, _ = research_view(md, cfg, holdout=True)
    assert full.calendar[-1] == md.calendar[-1]


def test_trials_count_distinct_combos(tmp_path):
    log = TrialLog(tmp_path / "trials.csv")
    m = {"start": "2020-01-01", "end": "2020-12-31", "days": 252, "sharpe": 0.5, "sharpe_daily": 0.03,
         "skew": 0, "kurt": 3, "cagr": 0.05, "max_dd": -0.1}
    log.log("p3", "D", {"lookback": 252}, "dev", True, m)
    log.log("p3", "D", {"lookback": 252}, "dev", False, m)  # same combo, costs off
    log.log("p4", "D", {"lookback": 202}, "dev", True, {**m, "sharpe_daily": 0.01})
    log.log("p4", "B", {"x": 1}, "dev", True, m)
    assert len(log.read()) == 4
    assert log.n_trials("D") == 2 and log.n_trials() == 3
    assert log.sharpe_variance("D") == pytest.approx(0.0002)
