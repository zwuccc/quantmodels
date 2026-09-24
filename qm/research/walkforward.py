"""Phase 4: sensitivity grid, walk forward, returns by year. Holdout never loaded.

For each strategy:
1. Run every grid combo (each lookback moved 20% down and up) over the whole
   development period. That's the sensitivity table.
2. Walk forward: for each test year, pick the combo with the best Sharpe on the
   years BEFORE it (expanding window), then run it fresh on that year alone.
   Train Sharpe comes from slicing the development run. That's the same as a run
   stopped at the train end, because the engine and strategies only look back
   (the look ahead check proves it).
3. Stitch the test years into one out of sample record and compare with SPY.
"""
from __future__ import annotations

import pandas as pd

from qm.config import results_dir
from qm.engine.metrics import by_year, sharpe, summarize
from qm.engine.backtest import BacktestResult
from qm.research.common import (ORDER, banner, dsr_for, fmt_table, grid_combos, log_run, load_market,
                                params_for, research_view, row_from, run_window, stitch, trial_log,
                                tunable, write_json)
from qm.strategies import build_targets, uses_single_stocks


def _stitch_results(parts: list[BacktestResult]) -> BacktestResult:
    """Glue test year results into one out of sample record."""
    r = stitch([x.returns for x in parts])
    cap = parts[0].initial_capital
    return BacktestResult(
        equity=(1 + r).cumprod() * cap, returns=r,
        trades=pd.concat([x.trades for x in parts], ignore_index=True),
        traded=stitch([x.traded for x in parts]), costs=stitch([x.costs for x in parts]),
        exposure=stitch([x.exposure for x in parts]), final_weights=parts[-1].final_weights,
        pending=parts[-1].pending, skipped=[s for x in parts for s in x.skipped], initial_capital=cap)


def run_walkforward(cfg: dict, names: list[str] | None = None) -> pd.DataFrame:
    names = names or ORDER
    md, splits = research_view(load_market(cfg), cfg)
    log = trial_log(cfg)
    out = results_dir(cfg) / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    windows = splits.wf_windows()
    summary, wf_rows, years, final = [], [], {}, {}

    for name in names:
        combos = grid_combos(cfg, name)
        default = params_for(cfg, name)
        grid_rows, dev_runs, targets = [], {}, {}
        for k, ov in enumerate(combos):
            p = params_for(cfg, name, ov)
            tg = build_targets(md, p)
            run = run_window(md, cfg, tg, splits.data_start, splits.dev_end)
            if not run:
                continue
            targets[k], dev_runs[k] = tg, run
            log_run(log, "phase4", name, p, "dev", run)
            is_default = all(p[key] == default[key] for key in ov)
            moved = {key: ("-20%" if v < default[key] else "+20%" if v > default[key] else "0")
                     for key, v in ov.items() if isinstance(v, (int, float))}
            grid_rows.append({"combo": k, **ov, "default": is_default,
                              "all_down": bool(moved) and all(x == "-20%" for x in moved.values()),
                              "all_up": bool(moved) and all(x == "+20%" for x in moved.values()),
                              "sharpe": run["m_on"]["sharpe"], "cagr": run["m_on"]["cagr"],
                              "max_dd": run["m_on"]["max_dd"], "spy_cagr": run["m_bench"]["cagr"],
                              "beats_spy_cagr": run["m_on"]["cagr"] > run["m_bench"]["cagr"]})
        if not dev_runs:
            print(f"{name}: no data in the development period")
            continue
        grid = pd.DataFrame(grid_rows)
        grid.to_csv(out / f"{name}_grid.csv", index=False)
        dk = int(grid.loc[grid["default"], "combo"].iloc[0]) if grid["default"].any() else int(grid["combo"].iloc[0])
        years[name] = by_year(dev_runs[dk]["on"].returns)
        years.setdefault("SPY", by_year(dev_runs[dk]["bench"].returns))

        # walk forward
        tests = []
        for w in windows:
            best, best_sr = None, -1e9
            for k, run in dev_runs.items():
                r = run["on"].returns
                r = r[r.index <= w["train_end"]]
                if len(r) < 252:
                    continue
                s = sharpe(r, cfg["engine"]["risk_free"])
                if s > best_sr:
                    best, best_sr = k, s
            if best is None:
                continue
            p = params_for(cfg, name, combos[best])
            test = run_window(md, cfg, targets[best], w["test_start"], w["test_end"], uses_single_stocks(p))
            if not test:
                continue
            log_run(log, "phase4", name, p, f"wf_test_{w['label']}", test)
            tests.append(test)
            wf_rows.append({"strategy": name, "test_year": w["label"], "picked": combos[best],
                            "train_sharpe": best_sr, "test_return": test["m_on"]["total_return"],
                            "spy_return": test["m_bench"]["total_return"]})
        if not tests:
            continue
        keys = ["on", "off", "bench"] + (["ew"] if "ew" in tests[0] else [])
        oos = {k: _stitch_results([t[k] for t in tests]) for k in keys}
        oos.update({f"m_{k}": summarize(oos[k], cfg["engine"]["risk_free"]) for k in keys})
        row = row_from(name, oos, cfg, dsr_for(log, name, oos["m_on"]))
        row["grid_sharpe_min"] = grid["sharpe"].min()
        row["grid_sharpe_max"] = grid["sharpe"].max()
        row["grid_share_beating_spy"] = grid["beats_spy_cagr"].mean()
        row["default_dev_sharpe"] = float(grid.loc[grid["combo"] == dk, "sharpe"].iloc[0])
        summary.append(row)
        pd.DataFrame({"strategy": oos["on"].returns, "strategy_no_costs": oos["off"].returns,
                      "spy": oos["bench"].returns}).to_csv(out / f"{name}_oos_returns.csv")

        best_dev = int(grid.sort_values("sharpe", ascending=False)["combo"].iloc[0])
        final[name] = {"default": tunable(default), "wf_pick": tunable(params_for(cfg, name, combos[best_dev]))}

    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(out / "walkforward_picks.csv", index=False)
    pd.DataFrame(years).to_csv(out / "by_year.csv")
    write_json(out / "final_settings.json", final)

    for line in banner(cfg, names):
        print(line)
    print(f"\nWalk forward out of sample {windows[0]['test_start'].date()} to {splits.dev_end.date()}, costs on")
    print(f"Holdout ({splits.holdout_start.date()} on) was NOT loaded.\n")
    if not df.empty:
        print(fmt_table(df, ["strategy", "cagr", "cagr_no_costs", "sharpe", "dsr_own", "dsr_all", "max_dd",
                             "spy_cagr", "gap_cagr_vs_spy", "ew_universe_cagr", "grid_sharpe_min", "grid_sharpe_max"]))
        for _, r in df[df["flags"] != ""].iterrows():
            print(f"TOO GOOD, check for bugs: {r['strategy']}: {r['flags']}")
    print(f"\nTrials logged so far: {log.n_trials()} distinct combos")
    return df
