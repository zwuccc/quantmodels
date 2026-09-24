# PLAN

A research tool that tests six stock trading strategies on free data and asks one
question: does any of them still beat buying SPY after costs, on data that was
never used to tune it?

"None of them do" is an acceptable answer. The goal is an honest test, not a
good looking backtest.

---

## 1. Time splits (fixed before any strategy is run)

All dates live in `config/default.yaml`. They are set once in Phase 1 when the
data snapshot is frozen, then never moved.

| Segment | Dates | Used for |
|---|---|---|
| Warmup | 2005-01-01 onward | Lookbacks only (200 day average, 12 month momentum, 8 quarters of SUE) |
| In sample | start to 2015-12-31 | Phase 3. Default settings only. Look, don't tune. |
| Walk forward test | 2016-01-01 to holdout start | Phase 4. Each test year uses settings picked only on earlier years. |
| Holdout | last 2 years of the snapshot (about 2024-09 to 2026-09) | Phase 5. Run once. Never tuned on. |

Why this shape: the in sample and walk forward test years don't overlap, so
what I see in Phase 3 can't leak into the Phase 4 test years.

Known weak spot: SEC XBRL data only starts around 2009 to 2011. Strategy E needs
12 quarters of EPS before its first signal, so E and F get only a few in sample
years (roughly 2012 to 2015). Their results will be noisier than the others.

### Holdout lock

- `qm/engine/splits.py` clips every data load at `holdout_start` by default.
- Loading holdout dates needs an explicit `--holdout` flag.
- The first holdout run writes `results/holdout_record.json` (date, git sha,
  settings used). A second run refuses to start while that file exists.

---

## 2. Folder layout

```
quantmodels/
  CLAUDE.md                 ground rules, commands, conventions
  PLAN.md                   this file
  pyproject.toml            deps and pytest config
  config/
    default.yaml            dates, costs, universe, strategy defaults, grids
  qm/
    config.py               load yaml into a typed Config object
    cli.py                  `python -m qm <command>`
    data/
      universe.py           S&P 500 + Nasdaq 100 members (Wikipedia snapshot) + ETF list
      prices.py             yfinance loader, stooq fallback, parquet cache
      edgar.py              ticker to CIK map, companyfacts download, rate limiter
      fundamentals.py       point in time tables and the as_of() lookup
      quality.py            gap, jump, duplicate checks; skip log
    engine/
      backtest.py           the daily loop (one engine for all six strategies)
      costs.py              commission + slippage model
      metrics.py            CAGR, Sharpe, drawdown, turnover, win rate, holding days
      stats.py              PSR and deflated Sharpe ratio
      splits.py             in sample, walk forward windows, holdout guard
      leakcheck.py          truncation test for look ahead
      trials.py             append every run to results/trials.csv
      sanity.py             "too good to be true" flags (rule 9)
    strategies/
      base.py               Strategy interface
      a_trend.py            200 day filter, 50/200 cross
      b_meanrev.py          RSI(2) pullback
      c_breakout.py         55/20 Donchian with ATR sizing
      d_momentum.py         12-1 month cross sectional momentum
      e_pead.py             SUE post earnings drift
      f_grossprof.py        gross profitability
    research/
      insample.py           Phase 3 runner
      walkforward.py        Phase 4 walk forward
      sensitivity.py        Phase 4 plus/minus 20% grid
      holdout.py            Phase 5 runner (with lock)
      report.py             Phase 6 HTML report
      signals.py            Phase 6 paper signal list
  tests/                    pytest suite (list in section 7)
  data/                     gitignored. raw downloads and processed tables
    raw/prices/{TICKER}.parquet
    raw/edgar/CIK##########.json.gz
    raw/universe/*.html
    processed/fundamentals_pit.parquet
    logs/skipped.csv        every name skipped and why (rule 8)
    manifest.json           what was downloaded, from where, when
  results/                  committed
    trials.csv              every parameter combination ever run (rule 4)
    phase3/ phase4/ phase5/ small CSV and JSON summaries
    report.html             Phase 6
    signals_latest.csv      Phase 6
```

---

## 3. Data flow

```
Wikipedia pages ──> universe.py ──> tickers.csv (current members, plus ETFs)
                                         │
      yfinance ──┐                       ▼
                 ├─> prices.py ──> raw/prices/*.parquet ──> quality.py ──> panel (date x ticker, OHLCV)
      stooq ─────┘   (fallback)                                  │
                                                                 ▼
sec.gov tickers ─> edgar.py ─> raw/edgar/*.json.gz ─> fundamentals.py ─> fundamentals_pit.parquet
                                                                 │        (every value keeps its filed date)
                                                                 ▼
                                    strategies/*.py  (signal from data up to close of day t)
                                                                 │  target weights
                                                                 ▼
                                    engine/backtest.py  (trades at open of day t+1, costs on)
                                                                 │
                                          ┌──────────────────────┼─────────────────────┐
                                          ▼                      ▼                     ▼
                                   metrics.py / stats.py    trials.csv           sanity.py flags
                                          │
                                          ▼
                                   research/*  ──> results/  ──> report.html + signals_latest.csv
```

### Prices

- yfinance first, `auto_adjust=True` (split and dividend adjusted OHLC). Stooq
  only if yfinance fails for that ticker.
- One source per ticker for its whole history. Never splice two sources.
- Cached as one parquet per ticker. A cached file is never re-downloaded unless
  `--refresh` is passed. `manifest.json` records source and download time.
- Ticker format: `BRK.B` becomes `BRK-B` for yfinance and `brk-b.us` for stooq.
- Quality checks: duplicate dates, non positive prices, one day moves above 50%
  that reverse (likely bad split adjustment), long gaps. A failed check means the
  ticker is skipped and logged in `data/logs/skipped.csv`. No filling.

### Fundamentals (SEC EDGAR)

- `company_tickers.json` maps ticker to CIK.
- `companyfacts/CIK##########.json` per company. Cached gzipped.
- Requests send a `User-Agent` with a name and email. A token bucket limits us
  to 8 requests per second (under SEC's 10).
- Tags used (first one present wins, logged which):
  - EPS: `EarningsPerShareDiluted`, then `EarningsPerShareBasic`
  - Revenue: `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`
  - Cost of revenue: `CostOfRevenue`, `CostOfGoodsAndServicesSold`, `CostOfGoodsSold`
  - Total assets: `Assets`
- Every row stored with: cik, ticker, tag, period start, period end, value,
  form, accession number, **filed date**.
- Point in time rule: for each (company, tag, period) keep the **first** filed
  value. Later restatements are ignored, because a trader on that day only saw
  the original number.
- `as_of(date)` returns only rows with `filed <= date`. Signals built on day t
  use `as_of(t)`, and the engine trades them at the open of t+1.
- Quarterly EPS: 3 month values from 10-Q. Q4 has no 10-Q, so Q4 EPS is
  derived as annual EPS minus the three quarters, dated by the 10-K filed date.
  This is an approximation (share counts change inside a year). I'll log how
  many Q4 values are derived and may drop them if they look noisy.

---

## 4. The engine

One engine for all six strategies. Strategies only produce signals.

**Input:** a target weight table (date x ticker), built from data up to the
close of each date.

- A number means "hold this fraction of equity".
- 0 means "exit".
- NaN means "leave this position alone".

The NaN rule lets event strategies (B, C, E) open one name without forcing a
rebalance of everything else. Monthly strategies (D, F) set every name on
rebalance days.

**Daily loop, for day t+1:**

1. Read the targets decided at the close of day t.
2. Work out the trades needed at the open of t+1 using current equity.
3. Fill at the adjusted open of t+1. Charge commission (10 bps) and slippage
   (5 bps) on traded value. Both come from config.
4. If a name has no open price on t+1, that trade is skipped and logged. If a
   held name's data ends, it's closed at its last close and logged.
5. Mark to market at the close of t+1.

**Rules:** long only, no leverage, no shorting. Cash earns 0% (configurable).
Weights above 100% total are scaled down. Every run is done twice, with and
without costs.

**Output:** daily equity, daily returns, a trade list (entry date, exit date,
return, holding days), and daily turnover.

**Benchmark:** SPY buy and hold over the exact same dates, paying the same cost
on its one entry trade.

---

## 5. The six strategies (defaults)

All long only. Universe notes are in the table.

| | Strategy | Universe | Signal | Exit / rebalance | Sizing | Source |
|---|---|---|---|---|---|---|
| A1 | 200 day trend | SPY | Close > 200d SMA: hold SPY | Close < 200d SMA: switch to IEF | 100% | common practice, Faber 2007 |
| A2 | 50/200 cross | SPY | 50d SMA > 200d SMA: hold SPY | Otherwise IEF | 100% | common practice |
| A3 | Multi asset trend | SPY, EFA, EEM, VNQ, GLD, DBC | Each asset above its 200d SMA | Its slice goes to IEF when below | equal slices | Faber 2007 |
| B | RSI(2) pullback | stocks | RSI(2) < 10 and close > 200d SMA | Close > 5d SMA, or 10 days | 10 slots, 10% each; lowest RSI first if more signals than slots | Connors |
| C | Donchian breakout | ETFs (default), stocks as a variant | Close at new 55 day high | Close at new 20 day low | risk 1% of equity per 1 ATR(20), cap 20% per name | Turtle rules |
| D | 12-1 momentum | stocks | Return from t-252 to t-21 | Monthly, last trading day | top 10%, equal weight | Jegadeesh & Titman 1993 |
| E | PEAD | stocks with XBRL EPS | SUE = (EPS_q - EPS_q-4) / std(last 8 such surprises) | Buy the open after the filing date, hold 60 trading days | top 10% of SUE vs trailing cross section, equal weight per position, max 5% | Bernard & Thomas 1989 |
| F | Gross profitability | stocks with XBRL data, financials dropped | (Revenue - COGS) / Total assets, latest annual values filed by t | Monthly | top 20%, equal weight | Novy Marx 2013 |

Notes:
- E's "top 10%" needs a cutoff that doesn't peek at future filings. I'll rank
  each new SUE against all SUEs filed in the trailing 365 days.
- F drops banks, insurers and REITs (they mostly lack a cost of revenue line),
  same as the paper. Missing COGS means skip and log, not zero.

---

## 6. Honesty tooling

**Trials log (rule 4).** Every backtest call through the research runners
appends one row to `results/trials.csv`: trial id, time, git sha, phase,
strategy, params (JSON), period, costs on/off, days, Sharpe, skew, kurtosis,
CAGR, max drawdown. Cost on/off of the same settings counts as one trial.

**Deflated Sharpe (Bailey & Lopez de Prado 2014).** Uses per period Sharpe,
skew, kurtosis, sample length, the variance of Sharpe across trials, and N =
number of distinct trials. I'll report it two ways: N for that strategy, and N
across all strategies (the stricter one, since in the end we pick a winner from
everything).

**Walk forward (Phase 4).** Expanding train window from 2005. For each test
year from 2016 on: pick the best settings from a small grid using only the
years before it, then run them on that one year. Stitch the test years into one
out of sample equity curve.

**Sensitivity (Phase 4).** Move each lookback 20% down and 20% up, one at a
time and all together. If the result falls apart, it was probably luck.

**Look ahead check.** For each strategy: build signals on the full data, then
on data cut off at a random day T. Signals up to T must match exactly. Any
difference means the strategy peeks.

**Too good flags (rule 9).** `sanity.py` flags Sharpe above 2, win rate above
70%, very smooth equity (R squared of log equity vs time above 0.98), or max
drawdown under 5% over 5+ years. A flag means I stop and hunt for a bug before
reporting. Checklist: look ahead test, fill timing, costs applied, bad prices
(huge one day jumps), survivorship, and whether the gain comes from a handful
of trades.

**Survivorship warning (rule 6).** Every report that touches single stocks
(B, C stock variant, D, E, F) prints a banner: the universe is today's index
members, so dead and delisted companies are missing, and results will look
better than reality. Momentum and PEAD are hit hardest.

---

## 7. Test list

Data (Phase 1)
- `test_fundamentals_pit.py`: a value filed on date X is invisible to `as_of(X - 1 day)` and visible to `as_of(X)`, and the earliest trade it can drive is the open of the next trading day after X.
- `test_fundamentals_pit.py`: a restated value does not replace the original filed value.
- `test_prices_cache.py`: a second load reads from disk and makes no network call (network mocked to raise).
- `test_prices_cache.py`: when yfinance fails, stooq is tried; when both fail the ticker is skipped and logged.
- `test_quality.py`: a series with a gap or a bad jump is skipped and logged, never filled.
- `test_edgar_rate_limit.py`: the limiter never exceeds 10 requests in any 1 second window; the User-Agent header is set.

Engine (Phase 2)
- `test_engine_hand.py`: 2 assets, 6 days, prices chosen so the answer can be checked on paper. Signal at the close of day 2 fills at the open of day 3. Checks cash, shares, costs and final equity to the cent.
- `test_engine_leak.py`: a strategy that uses tomorrow's close on purpose. The look ahead check must catch it (the test asserts that it raises).
- `test_engine_timing.py`: a signal on day t never changes the return of day t.
- `test_costs.py`: cost equals traded value times 15 bps; zero cost mode matches.
- `test_nan_targets.py`: NaN leaves a position alone, 0 exits it.
- `test_metrics.py`: CAGR, max drawdown, Sharpe on tiny known series.
- `test_dsr.py`: DSR with N = 1 equals the probabilistic Sharpe ratio; DSR falls as N rises; check against a worked example from the paper if I can confirm the numbers.
- `test_splits.py`: holdout dates can't be loaded without the flag; a second holdout run is refused.
- `test_trials.py`: every runner call adds exactly one row.

Strategies (Phase 3)
- One small test per signal on a toy series: RSI(2), SMA cross, Donchian high/low, ATR, 12-1 ranking, SUE, gross profitability.
- The look ahead check run on all six real strategies.

---

## 8. Phases

| Phase | Build | Output | Then |
|---|---|---|---|
| 0 | PLAN.md, CLAUDE.md | this file | stop |
| 1 | data layer, caching, PIT fundamentals, data tests | data snapshot, coverage summary, skip log | stop |
| 2 | engine, metrics, DSR, leak check, engine tests | tests passing | stop |
| 3 | six strategies with defaults, in sample only | one summary table vs SPY | stop |
| 4 | walk forward, sensitivity grids, returns by year | trials.csv, stress tables | stop |
| 5 | holdout run, once | holdout table, reported as is | stop |
| 6 | HTML report, signal list | report.html, signals_latest.csv | stop |

---

## 9. Budget habits

- Never print a large file. Inspect with `head`, `.shape`, `.describe()`.
- Every runner writes a small summary CSV and prints at most a few dozen lines.
- Downloads are cached. A rerun after the first download costs no network.
- Long downloads run in the background with a progress count, not a stream of logs.

---

## 10. Open questions and things I'm unsure about

1. **Network access.** Yahoo, Stooq, SEC and Wikipedia are blocked in this
   environment right now. Phase 1 can't run until they are allowed.
2. **Strategy C universe.** The prompt doesn't say. Turtle rules were built for
   a mix of futures markets, so ETFs are the closer match and avoid
   survivorship bias. Default: ETFs, with a stock version as a counted variant.
3. **Cash rate.** Cash earns 0% and Sharpe uses a 0% risk free rate. This is
   simple and a bit harsh on strategies that sit in cash (B, C). A T-bill rate
   could be added later.
4. **Q4 EPS** is derived (annual minus three quarters). It's the standard trick
   but it's noisy.
5. **XBRL start date** means E and F have much shorter histories than A to D.
6. **Current members only.** Survivorship bias can't be fixed with free data.
   The warning banner is the best I can do. Results for stock strategies should
   be read as an upper bound.
7. **Rebalance timing for A.** The prompt says daily checks. Monthly checks
   (Faber's version) trade less. I'll use daily as the default and test monthly
   as a counted variant in Phase 4.
