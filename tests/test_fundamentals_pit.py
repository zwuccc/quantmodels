"""Rule 1 for fundamentals: a value counts only from its SEC filed date."""
import pandas as pd
import pytest

from qm.data.fundamentals import Fundamentals, build_pit, extract_rows, sue_events, annual_gross_profitability
from qm.engine.backtest import run_backtest


def fact(start, end, val, form, filed, accn):
    d = {"end": end, "val": val, "form": form, "filed": filed, "accn": accn}
    if start:
        d["start"] = start
    return d


FACTS = {"facts": {"us-gaap": {
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        fact("2020-01-01", "2020-03-31", 1.00, "10-Q", "2020-05-06", "q1"),
        fact("2020-01-01", "2020-03-31", 1.50, "10-Q/A", "2020-08-03", "q1a"),   # restatement
        fact("2020-04-01", "2020-06-30", 1.10, "10-Q", "2020-08-05", "q2"),
        fact("2020-07-01", "2020-09-30", 1.20, "10-Q", "2020-11-04", "q3"),
        fact("2020-01-01", "2020-12-31", 4.80, "10-K", "2021-02-24", "k1"),
        fact("2020-01-01", "2020-06-30", 2.10, "10-Q", "2020-08-05", "q2ytd"),  # 6 month YTD, ignored
    ]}},
    "Revenues": {"units": {"USD": [fact("2020-01-01", "2020-12-31", 1000.0, "10-K", "2021-02-24", "k1")]}},
    "CostOfRevenue": {"units": {"USD": [fact("2020-01-01", "2020-12-31", 600.0, "10-K", "2021-02-24", "k1")]}},
    "Assets": {"units": {"USD": [fact(None, "2020-12-31", 2000.0, "10-K", "2021-02-24", "k1")]}},
}}}


@pytest.fixture
def fund(cfg):
    return Fundamentals(build_pit(extract_rows(FACTS, "XYZ", cfg), cfg))


def eps_q1(df):
    return df[(df["concept"] == "eps") & (df["end"] == "2020-03-31") & (df["period"] == "Q")]


def test_value_filed_on_x_is_invisible_the_day_before(fund):
    x = pd.Timestamp("2020-05-06")
    assert eps_q1(fund.as_of(x - pd.Timedelta(days=1))).empty
    assert eps_q1(fund.as_of(x))["value"].tolist() == [1.00]


def test_period_end_date_does_not_make_a_value_visible(fund):
    # The quarter ended 2020-03-31, but nothing was known until the filing on 2020-05-06.
    assert eps_q1(fund.as_of("2020-04-15")).empty


def test_restatement_does_not_replace_first_filed_value(fund):
    assert eps_q1(fund.as_of("2022-01-01"))["value"].tolist() == [1.00]


def test_q4_is_derived_from_annual_and_dated_by_10k(fund):
    q4 = fund.table[(fund.table["concept"] == "eps") & fund.table["derived"]]
    assert len(q4) == 1
    assert q4["value"].iloc[0] == pytest.approx(4.80 - 1.00 - 1.10 - 1.20)
    assert q4["filed"].iloc[0] == pd.Timestamp("2021-02-24")
    assert fund.as_of("2021-02-23")["derived"].sum() == 0


def test_ytd_values_are_not_treated_as_quarters(fund):
    q = fund.table[(fund.table["concept"] == "eps") & (fund.table["period"] == "Q")]
    assert pd.Timestamp("2020-06-30") in set(q["end"]) and 2.10 not in set(q["value"])


def test_gross_profitability_dated_by_filing(fund):
    gp = annual_gross_profitability(fund)
    assert gp["gpa"].iloc[0] == pytest.approx((1000 - 600) / 2000)
    assert gp["filed"].iloc[0] == pd.Timestamp("2021-02-24")
    assert annual_gross_profitability(fund.clip("2021-02-23")).empty


def test_earliest_trade_is_next_open_after_filing():
    # A signal from a filing on X is set at the close of X and fills at the open of X+1.
    d = pd.bdate_range("2021-02-22", periods=5)  # Mon..Fri, X = Wed 2021-02-24
    px = pd.DataFrame({"XYZ": [10, 11, 12, 13, 14]}, index=d, dtype=float)
    tgt = pd.DataFrame({"XYZ": [1.0]}, index=[pd.Timestamp("2021-02-24")])
    res = run_backtest(px, px, tgt, 0.0, 1.0)
    first_fill = res.exposure[res.exposure > 0].index[0]
    assert first_fill == pd.Timestamp("2021-02-25")


def test_sue_uses_only_filed_data():
    rows = []
    for i, y in enumerate(range(2010, 2016)):
        for q, (s, e) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]):
            end = pd.Timestamp(f"{y}-{e}")
            rows.append({"ticker": "XYZ", "concept": "eps", "tag": "t", "start": pd.Timestamp(f"{y}-{s}"), "end": end,
                         "days": 90, "period": "Q", "value": 1.0 + 0.1 * i + 0.03 * ((i * 4 + q) % 3),
                         "form": "10-Q", "accn": "", "filed": end + pd.Timedelta(days=40), "derived": False})
    f = Fundamentals(pd.DataFrame(rows))
    ev = sue_events(f, 8)
    assert len(ev) > 0
    assert (ev["filed"] >= ev["end"] + pd.Timedelta(days=40)).all()
    # clipping removes later events and does not change earlier ones
    cut = ev["filed"].iloc[len(ev) // 2]
    ev2 = sue_events(f.clip(cut), 8)
    pd.testing.assert_frame_equal(ev[ev["filed"] <= cut].reset_index(drop=True), ev2.reset_index(drop=True))


def test_row_filed_before_its_period_ended_is_dropped(cfg):
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        fact(None, "2009-12-31", 100.0, "10-Q", "2009-11-02", "typo")]}}}}}
    assert build_pit(extract_rows(facts, "ROP", cfg), cfg).empty


def test_gross_profitability_outside_bounds_is_dropped(fund):
    assert len(annual_gross_profitability(fund, (-1.0, 3.0))) == 1
    assert annual_gross_profitability(fund, (-1.0, 0.1)).empty   # 0.2 is outside


def _quarters(late_end=None, late_filed=None):
    rows = []
    for i, y in enumerate(range(2010, 2016)):
        for q, (s, e) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]):
            end = pd.Timestamp(f"{y}-{e}")
            filed = end + pd.Timedelta(days=40)
            if late_end is not None and end == pd.Timestamp(late_end):
                filed = pd.Timestamp(late_filed)  # first reported much later, as a comparative
            rows.append({"ticker": "XYZ", "concept": "eps", "tag": "t", "start": pd.Timestamp(f"{y}-{s}"), "end": end,
                         "days": 90, "period": "Q", "value": 1.0 + 0.1 * i + 0.03 * ((i * 4 + q) % 3),
                         "form": "10-Q", "accn": "", "filed": filed, "derived": False})
    return Fundamentals(pd.DataFrame(rows))


def test_sue_old_quarter_reported_late_does_not_change_earlier_events():
    # Real bug (TPR 2012): a quarter first reported in a later filing made the
    # full history skip an event that was visible at the time.
    f = _quarters(late_end="2011-06-30", late_filed="2014-03-01")
    ev = sue_events(f, 8)
    for cut in ["2013-02-15", "2013-08-15", "2014-02-15"]:
        part = sue_events(f.clip(cut), 8)
        full = ev[ev["filed"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, part.reset_index(drop=True), check_dtype=False)


def test_gross_profit_tag_filed_later_does_not_hide_earlier_version(cfg):
    # Real bug (AXON 2013): GrossProfit first filed later than revenue and cost.
    early = {"start": "2012-01-01", "end": "2012-12-31", "form": "10-K", "filed": "2013-02-20", "accn": "k12"}
    late = {**early, "filed": "2014-02-20", "accn": "k13"}
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [{**early, "val": 1000.0}]}},
        "CostOfRevenue": {"units": {"USD": [{**early, "val": 600.0}]}},
        "GrossProfit": {"units": {"USD": [{**late, "val": 400.0}]}},
        "Assets": {"units": {"USD": [{"end": "2012-12-31", "val": 2000.0, "form": "10-K", "filed": "2013-02-20", "accn": "k12"}]}},
    }}}
    f = Fundamentals(build_pit(extract_rows(facts, "XYZ", cfg), cfg))
    cut = pd.Timestamp("2013-06-30")
    full = annual_gross_profitability(f)
    part = annual_gross_profitability(f.clip(cut))
    pd.testing.assert_frame_equal(full[full["filed"] <= cut].reset_index(drop=True), part.reset_index(drop=True), check_dtype=False)
    assert len(part) == 1 and part["filed"].iloc[0] == pd.Timestamp("2013-02-20")
