# Phase 3 sanity checks (rule 9)

## Look ahead check on real data
The first real run FAILED for E and F. Both were real leaks, both fixed:
- E: an old quarter first reported in a later filing made the full history skip
  an event that was visible at the time (TPR, Feb 2012).
- F: a GrossProfit tag first filed after revenue and cost hid the earlier
  revenue minus cost version (AXON, 2013).
After the fix, E and F pass at 6 cut dates from 2011 to 2022. All nine pass in Phase 3.

## F flagged: win rate 86% (limit 70%)
Checked with a placebo: same F code, gross profitability shuffled at random, 12 runs.
- Placebo win rate: mean 79%, range 75% to 83%. So the high win rate comes from
  a 2010 to 2015 bull market, ~335 day holds and a survivor universe, not a bug.
- Real F CAGR 20.3% vs placebo mean 18.5% (range 16.0% to 20.5%); beats 11 of 12.
- Top 10 of 50 trades made 67% of the profit (normal skew for stocks).
Conclusion: not a bug. A small in sample edge over random picks, on survivor data.

## Survivorship, measured
Equal weight of today's members, 2005 to 2015: 15.0% a year vs SPY 7.6%.
