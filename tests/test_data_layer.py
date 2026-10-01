import numpy as np
import pandas as pd
import pytest

from qm.data.edgar import RateLimiter
from qm.data.prices import PriceStore, build_panel
from qm.data.quality import check_prices


def ohlcv(n=300, start="2020-01-01", seed=0):
    d = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=d)


def no_network(t, s):
    raise AssertionError("network used")


def test_second_load_comes_from_cache(cfg):
    calls = []

    def fake(t, s):
        calls.append(t)
        return ohlcv()

    s1 = PriceStore(cfg, fetchers=[("fake", fake)], sleep=0)
    s1.get("AAA")
    s2 = PriceStore(cfg, fetchers=[("boom", no_network)], sleep=0)
    df = s2.get("AAA")
    assert calls == ["AAA"] and len(df) == 300 and s2.network_calls == 0


def test_fallback_then_skip_and_log(cfg):
    def bad(t, s):
        raise ValueError("down")

    store = PriceStore(cfg, fetchers=[("yf", bad), ("stooq", lambda t, s: ohlcv())], sleep=0)
    assert store.get("AAA") is not None
    store2 = PriceStore(cfg, fetchers=[("yf", bad), ("stooq", bad)], sleep=0)
    assert store2.get("BBB") is None
    log = pd.read_csv(store2.skiplog.path)
    assert log["name"].tolist() == ["BBB"] and "stooq" in log["reason"].iloc[0]


def test_gap_is_skipped_and_logged_never_filled(cfg):
    cfg["data_quality"]["max_internal_gaps"] = 0
    good, gappy = ohlcv(), ohlcv(seed=1).drop(ohlcv().index[150])
    data = {"SPY": good, "AAA": good, "GAP": gappy}
    store = PriceStore(cfg, fetchers=[("fake", lambda t, s: data[t])], sleep=0)
    for t in data:
        store.get(t)
    panel = build_panel(store, ["AAA", "GAP"], "SPY")
    assert list(panel["close"].columns) == ["AAA"]
    log = pd.read_csv(store.skiplog.path)
    assert "GAP" in log["name"].tolist() and "gap" in log["reason"].iloc[-1]


def test_quality_catches_bad_jump():
    df = ohlcv()
    df.iloc[100, :4] *= 3  # a one day spike that reverses
    assert "jump" in check_prices(df, None, {"min_history_days": 10, "max_internal_gaps": 0,
                                             "bad_jump": 0.5, "bad_jump_reversal": 0.3})


def test_rate_limiter_never_exceeds_limit():
    t = [0.0]
    stamps = []

    def clock():
        return t[0]

    def sleep(s):
        t[0] += s

    rl = RateLimiter(8, clock=clock, sleep=sleep)
    for _ in range(50):
        rl.wait()
        stamps.append(t[0])
        t[0] += 0.01
    stamps = np.array(stamps)
    worst = max(((stamps >= s) & (stamps < s + 1.0)).sum() for s in stamps)
    assert worst <= 8


def test_edgar_sends_user_agent(cfg, monkeypatch):
    from qm.data.edgar import EdgarClient

    seen = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}

    class FakeSession:
        def get(self, url, headers, timeout):
            seen.update(headers)
            return FakeResp()

    monkeypatch.setenv("QM_SEC_USER_AGENT", "Test User test@example.com")
    c = EdgarClient(cfg, session=FakeSession())
    assert c.ticker_map() == {"AAPL": 320193}
    assert seen["User-Agent"] == "Test User test@example.com"


def test_edgar_refuses_without_user_agent(cfg, monkeypatch):
    from qm.data.edgar import EdgarClient

    monkeypatch.delenv("QM_SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        EdgarClient(cfg)


Q = {"min_history_days": 10, "max_internal_gaps": 0, "bad_jump": 0.5, "bad_jump_reversal": 0.3,
     "min_clean_run": 60, "big_distribution": 0.10, "max_event_move": 0.25}


def test_leading_copied_prices_are_cut_and_inner_ones_are_gaps():
    from qm.data.quality import trim_leading_stale
    df = ohlcv(400)
    df.iloc[1:100, df.columns.get_loc("close")] = df["close"].iloc[0]   # fake flat history
    df.iloc[:100, df.columns.get_loc("volume")] = 0
    out, filled, cut = trim_leading_stale(df, 60)
    assert cut == 100 and filled == 0 and out.index[0] == df.index[100]
    df.iloc[250, df.columns.get_loc("close")] = df["close"].iloc[249]   # one copied day inside real trading
    df.iloc[250, df.columns.get_loc("volume")] = 0
    _, filled, _ = trim_leading_stale(df, 60)
    assert filled == 1


def test_broken_spin_off_adjustment_is_caught():
    # DHR style: Yahoo books a spin off as a huge "dividend" and the adjusted
    # series jumps +61% while the stock itself barely moved.
    from qm.data.quality import check_adjustments
    adj = ohlcv(300)
    adj.iloc[:150, :4] /= 1.61
    unadj = pd.DataFrame({"close_raw": adj["close"] * 1.0, "adj_close": adj["close"],
                          "dividends": 0.0, "splits": 0.0}, index=adj.index)
    unadj.iloc[:150, 0] = adj["close"].iloc[:150] * 1.61
    unadj.iloc[150, unadj.columns.get_loc("dividends")] = 0.35 * unadj["close_raw"].iloc[149]
    assert "broken adjustment" in check_adjustments(adj, unadj, Q)


def test_correct_special_dividend_adjustment_passes():
    from qm.data.quality import check_adjustments
    adj = ohlcv(300)
    unadj = pd.DataFrame({"close_raw": adj["close"], "adj_close": adj["close"], "dividends": 0.0, "splits": 0.0}, index=adj.index)
    unadj.iloc[150, unadj.columns.get_loc("dividends")] = 0.2 * unadj["close_raw"].iloc[149]
    assert check_adjustments(adj, unadj, Q) is None
    assert "no unadjusted" in check_adjustments(adj, None, Q)


def test_a_few_copied_days_become_blanks_not_prices(cfg):
    good = ohlcv()
    copied = good.copy()
    copied.iloc[150, copied.columns.get_loc("close")] = copied["close"].iloc[149]
    copied.iloc[150, copied.columns.get_loc("volume")] = 0
    data = {"SPY": good, "AAA": copied}
    store = PriceStore(cfg, fetchers=[("fake", lambda t, s: data[t])], sleep=0)
    for t in data:
        store.get(t)
    panel = build_panel(store, ["AAA"], "SPY")
    assert np.isnan(panel["close"]["AAA"].iloc[150]) and panel["close"]["AAA"].notna().sum() == 299
