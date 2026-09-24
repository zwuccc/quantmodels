"""Strategy registry. Every strategy is fn(MarketData, params) -> target weights."""
from qm.strategies.a_trend import trend_cross, trend_multi, trend_sma
from qm.strategies.b_meanrev import meanrev
from qm.strategies.c_breakout import breakout
from qm.strategies.d_momentum import momentum
from qm.strategies.e_pead import pead
from qm.strategies.f_grossprof import grossprof

KINDS = {
    "trend_sma": trend_sma, "trend_cross": trend_cross, "trend_multi": trend_multi,
    "meanrev": meanrev, "breakout": breakout, "momentum": momentum,
    "pead": pead, "grossprof": grossprof,
}

USES_STOCKS = {"meanrev", "momentum", "pead", "grossprof"}


def build_targets(md, params: dict):
    return KINDS[params["kind"]](md, params)


def uses_single_stocks(params: dict) -> bool:
    return params["kind"] in USES_STOCKS or params.get("universe") == "stocks"
