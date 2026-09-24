"""python -m qm <command>. Add --synthetic to any command to use the fake market."""
from __future__ import annotations

import argparse

import pandas as pd

from qm.config import data_dir, load_config


def cmd_universe(cfg, a):
    from qm.data.universe import build_universe
    u = build_universe(cfg, refresh=a.refresh)
    print(u["kind"].value_counts().to_string())


def cmd_prices(cfg, a):
    from qm.data.prices import PriceStore
    uni = pd.read_csv(data_dir(cfg) / "processed" / "tickers.csv")
    store = PriceStore(cfg)
    print(store.download_all(uni["ticker"].tolist(), refresh=a.refresh))


def cmd_panel(cfg, a):
    from qm.data.prices import PriceStore, build_panel
    from qm.research.common import panel_path
    uni = pd.read_csv(data_dir(cfg) / "processed" / "tickers.csv")
    store = PriceStore(cfg, fetchers=[])  # offline: cache only
    panel = build_panel(store, uni["ticker"].tolist(), cfg["universe"]["benchmark"])
    for f, df in panel.items():
        df.to_parquet(panel_path(cfg, f))
    c = panel["close"]
    print(f"panel: {c.shape[0]} days x {c.shape[1]} tickers, {c.index[0].date()} to {c.index[-1].date()}")


def cmd_edgar(cfg, a):
    from qm.data.edgar import EdgarClient
    uni = pd.read_csv(data_dir(cfg) / "processed" / "tickers.csv")
    client = EdgarClient(cfg)
    cmap = client.ticker_map(refresh=a.refresh)
    stocks = uni.loc[uni["kind"] == "stock", "ticker"]
    done = 0
    for i, t in enumerate(stocks, 1):
        if t not in cmap:
            client.skiplog.log(t, "edgar_cik", "no CIK in SEC ticker map")
            continue
        if client.companyfacts(cmap[t], refresh=a.refresh) is not None:
            done += 1
        if i % 50 == 0:
            print(f"  {i}/{len(stocks)}", flush=True)
    print(f"companyfacts cached for {done} of {len(stocks)} stocks")


def cmd_fundamentals(cfg, a):
    from qm.data.edgar import load_cached_facts
    from qm.data.fundamentals import build_pit, extract_rows, pit_path
    from qm.data.skiplog import skiplog_for
    import json
    uni = pd.read_csv(data_dir(cfg) / "processed" / "tickers.csv")
    cmap_path = data_dir(cfg) / "raw" / "edgar" / "company_tickers.json"
    raw = json.loads(cmap_path.read_text())
    cmap = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}
    log = skiplog_for(cfg)
    rows = []
    for t in uni.loc[uni["kind"] == "stock", "ticker"]:
        facts = load_cached_facts(cfg, cmap[t]) if t in cmap else None
        if facts is None:
            log.log(t, "fundamentals", "no cached companyfacts")
            continue
        r = extract_rows(facts, t, cfg)
        if r.empty:
            log.log(t, "fundamentals", "no usable XBRL tags")
            continue
        rows.append(r)
    pit = build_pit(pd.concat(rows, ignore_index=True), cfg)
    pit.to_parquet(pit_path(cfg))
    print(pit.groupby(["concept", "period"]).size().to_string())
    print(f"derived Q4 EPS rows: {int(pit['derived'].sum())}")


def cmd_synthetic(cfg, a):
    from qm.synthetic import generate
    if not cfg["synthetic"]:
        raise SystemExit("use: python -m qm --synthetic synthetic")
    print(generate(cfg))


def cmd_insample(cfg, a):
    from qm.research.insample import run_insample
    run_insample(cfg, a.strategy)


def cmd_walkforward(cfg, a):
    from qm.research.walkforward import run_walkforward
    run_walkforward(cfg, a.strategy)


def cmd_holdout(cfg, a):
    from qm.research.holdout import run_holdout
    run_holdout(cfg, confirm=a.holdout)


def cmd_report(cfg, a):
    from qm.research.report import build_report
    print(build_report(cfg))


def cmd_signals(cfg, a):
    from qm.research.signals import latest_signals
    latest_signals(cfg)


COMMANDS = {
    "universe": cmd_universe, "prices": cmd_prices, "panel": cmd_panel, "edgar": cmd_edgar,
    "fundamentals": cmd_fundamentals, "synthetic": cmd_synthetic, "insample": cmd_insample,
    "walkforward": cmd_walkforward, "sensitivity": cmd_walkforward, "holdout": cmd_holdout,
    "report": cmd_report, "signals": cmd_signals,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qm")
    ap.add_argument("--synthetic", action="store_true", help="use the fake market under synthetic/")
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--strategy", nargs="*", help="e.g. --strategy A1 D")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--holdout", action="store_true", help="confirm the one time holdout run")
    a = ap.parse_args(argv)
    cfg = load_config(synthetic=a.synthetic)
    COMMANDS[a.command](cfg, a)
