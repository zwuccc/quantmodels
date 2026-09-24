# quantmodels

Tests six stock trading strategies on free data and reports, honestly, whether
any of them beat buying SPY after costs, on data never used to tune them.

- Design: [PLAN.md](PLAN.md). Ground rules and conventions: [CLAUDE.md](CLAUDE.md).
- What the final report looks like (on FAKE data, numbers mean nothing):
  [demo/synthetic_report.html](demo/synthetic_report.html)

Research only. No broker code, no orders.

## Status

All code is written and tested (`pytest -q`, 53 tests, no network). It has
only been run end to end on a fake random walk market, because the build
environment couldn't reach Yahoo, Stooq, SEC or Wikipedia. **There are no real
results yet.**

## Run it on real data

Needs network access to: `query1.finance.yahoo.com`, `query2.finance.yahoo.com`,
`fc.yahoo.com`, `finance.yahoo.com`, `stooq.com`, `data.sec.gov`, `www.sec.gov`,
`en.wikipedia.org`.

```bash
pip install -e ".[dev]"
pytest -q

# SEC asks for a real name and email in the User-Agent
export QM_SEC_USER_AGENT="Your Name you@example.com"

# Phase 1: data (cached under data/, reruns cost nothing)
python -m qm universe          # today's S&P 500 + Nasdaq 100 + ETFs
python -m qm prices            # about 550 tickers, several minutes
python -m qm panel             # quality checks, builds the price panel
python -m qm edgar             # companyfacts, about 1 minute at 8 req/s
python -m qm fundamentals      # point in time table

# Then set dates.snapshot_end in config/default.yaml to the last close, and commit.
# That freezes the holdout (the 2 years before snapshot_end).

python -m qm insample          # Phase 3
python -m qm walkforward       # Phase 4 (sensitivity grid + walk forward)
python -m qm holdout --holdout # Phase 5, works exactly once
python -m qm signals           # Phase 6
python -m qm report            # Phase 6 -> results/report.html
```

If Wikipedia is unreachable, put a CSV with columns `ticker,sector` at
`data/raw/universe/tickers_manual.csv` and `universe` will use it.

Skipped names and the reason are in `data/logs/skipped.csv`.
Every setting ever tried is in `results/trials.csv`.

## Try the whole pipeline on fake data

```bash
python -m qm --synthetic synthetic
python -m qm --synthetic insample
python -m qm --synthetic walkforward
python -m qm --synthetic holdout --holdout
python -m qm --synthetic signals
python -m qm --synthetic report     # synthetic/results/report.html
```

On fake data nothing should beat the baseline. If something does, that's a bug.
