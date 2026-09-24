# CLAUDE.md

Research tool that backtests six trading strategies on free data and reports,
honestly, whether any beat SPY buy and hold after costs and out of sample.
See PLAN.md for the design.

## Ground rules (these beat everything else)

1. **No look ahead.** A signal built from the close of day t trades at the open
   of day t+1. Fundamentals count only from their SEC **filed date**, never the
   period end date.
2. **Costs are always on.** Default 10 bps per side commission plus 5 bps
   slippage, set in `config/default.yaml`. Show results with and without costs.
3. **Split the data.** Tune on older years. Test with walk forward windows. The
   last 2 years are a locked holdout. Touch it once, in Phase 5, and never tune
   on it.
4. **Count every test.** Every parameter combination tried goes into
   `results/trials.csv`. Report the deflated Sharpe ratio (Bailey and Lopez de
   Prado) using that count.
5. **Always compare against SPY buy and hold** over the same dates.
6. **Be upfront about survivorship bias.** The stock universe is today's index
   members. Every report that uses single stocks prints a warning.
7. **No live trading.** No broker APIs, no order code. Output is research and a
   paper signal list only.
8. **Missing data means skip the name and log it** to `data/logs/skipped.csv`.
   Never fill gaps with guesses (no ffill of prices, no zero for missing
   fundamentals).
9. **Too good means bug first.** Sharpe above 2, a very smooth equity curve, or
   a huge win rate: assume a bug, go find it, and write down what was checked.

## Working in phases

Work stops at the end of each phase. Summarize in plain words what was built
and what it found, then wait for the user to say go. Current phase is tracked
at the bottom of this file.

## Budget habits

- Never read large data files into context. Inspect with short scripts:
  `head`, `.shape`, `.describe()`, `.value_counts().head()`.
- Every download is cached under `data/`. Reruns must cost no network.
- Runners print short summaries (a few dozen lines max) and write detail to
  CSV/JSON under `results/`.
- Run long downloads in the background and check a progress count.

## Commands

```bash
pip install -e ".[dev]"                   # install
pytest -q                                 # all tests
pytest -q tests/test_engine_hand.py       # one file

python -m qm universe                     # build ticker list (cached)
python -m qm prices [--refresh]           # download or load prices (cached)
python -m qm edgar [--refresh]            # download companyfacts (cached, rate limited)
python -m qm fundamentals                 # build point in time table
python -m qm insample --strategy D        # Phase 3
python -m qm walkforward --strategy D     # Phase 4
python -m qm sensitivity --strategy D     # Phase 4
python -m qm holdout                      # Phase 5, runs once, needs --holdout
python -m qm report                       # Phase 6
python -m qm signals                      # Phase 6
```

(Commands are the plan; they get built phase by phase.)

## Conventions

- Python 3.11, package `qm/`. pandas and numpy. Parquet (pyarrow) for tables.
- All settings (dates, costs, universe, strategy defaults, grids) live in
  `config/default.yaml`. No magic numbers in strategy code.
- Price panels are wide DataFrames: index = trading date, columns = ticker.
- Strategies return a target weight DataFrame from data up to the close of each
  date. The engine alone shifts to the next open. Strategies never shift
  forward themselves. NaN = leave position alone, 0 = exit.
- Fundamentals are read only through `fundamentals.as_of(date)`.
- Every research run goes through a runner that logs to `results/trials.csv`.
  Unit tests write trials to a temp path, never the real log.
- The holdout is clipped out of every data load unless `--holdout` is passed.
- EDGAR requests send a User-Agent with name and email and stay under 8
  requests per second.
- `data/` is gitignored. `results/` is committed.
- Long only, no leverage. Cash earns 0% unless config says otherwise.
- Tests use tiny synthetic data and never touch the network.

## Status

- Phase 0 (plan): done
- Phase 1 (data): not started. Blocked until the network policy allows
  finance.yahoo.com hosts, stooq.com, sec.gov hosts, en.wikipedia.org.
