"""Phase 6: one self contained HTML report.

Reads the saved phase outputs, re-runs each strategy's DEFAULT settings over the
whole allowed period for the equity chart (no tuning happens here), and writes
results/report.html.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from qm.config import results_dir
from qm.config import cost_rate
from qm.engine.backtest import buy_and_hold_targets, run_backtest
from qm.engine.metrics import by_year
from qm.engine.splits import holdout_used
from qm.research.common import (ORDER, SURVIVORSHIP_WARNING, SYNTHETIC_WARNING, load_market, params_for,
                                research_view, run_window)
from qm.strategies import build_targets, uses_single_stocks

MAIN = ["A1", "B", "C", "D", "E", "F"]          # one line each on the main chart
VARIANTS = {"A2": "A1", "A3": "A1", "C_stocks": "C"}  # small multiples, family color
NAMES = {
    "A1": "A1 · 200 day trend (SPY/IEF)", "A2": "A2 · 50/200 cross (SPY/IEF)",
    "A3": "A3 · Multi asset trend", "B": "B · RSI(2) mean reversion", "C": "C · Breakout on ETFs",
    "C_stocks": "C · Breakout on stocks", "D": "D · 12 minus 1 momentum", "E": "E · Earnings drift (PEAD)",
    "F": "F · Gross profitability",
}
SLOT = {"A1": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

METRICS = [  # (column, label, format)
    ("cagr", "CAGR after costs", "pct"), ("cagr_no_costs", "CAGR before costs", "pct"),
    ("sharpe", "Sharpe", "num"), ("dsr_own", "Deflated Sharpe (own trials)", "num"),
    ("dsr_all", "Deflated Sharpe (all trials)", "num"), ("max_dd", "Max drawdown", "pct"),
    ("worst_year", "Worst year", "pct"), ("turnover", "Turnover per year", "x"),
    ("win_rate", "Win rate", "pct"), ("avg_hold_days", "Avg holding days", "int"),
    ("total_return", "Total return after costs", "pct"),
    ("total_return_no_costs", "Total return before costs", "pct"),
    ("exposure", "Avg invested", "pct"),
    ("spy_cagr", "SPY CAGR, same dates", "pct"), ("gap_cagr_vs_spy", "Gap vs SPY (CAGR)", "gap"),
    ("gap_sharpe_vs_spy", "Gap vs SPY (Sharpe)", "gapnum"),
    ("ew_universe_cagr", "Equal weight universe CAGR", "pct"), ("gap_cagr_vs_ew", "Gap vs equal weight", "gap"),
]


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or (isinstance(v, str) and not v):
        return "–"
    v = float(v)
    return {"pct": f"{v:.1%}", "num": f"{v:.2f}", "x": f"{v:.1f}×", "int": f"{v:.0f}",
            "gap": f"{v:+.1%}", "gapnum": f"{v:+.2f}"}[kind]


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _verdict(oos: pd.Series | None, hold: pd.Series | None) -> tuple[str, str]:
    if oos is None:
        return "not run", "No walk forward result yet."
    beat_oos = oos["gap_cagr_vs_spy"] > 0 and oos["gap_sharpe_vs_spy"] > 0
    strong = np.isfinite(oos.get("dsr_own", np.nan)) and oos["dsr_own"] >= 0.95
    why = [f"Out of sample it {'beat' if beat_oos else 'did not beat'} SPY after costs "
           f"({oos['gap_cagr_vs_spy']:+.1%} a year, Sharpe {oos['sharpe']:.2f} vs {oos['spy_sharpe']:.2f})."]
    why.append(f"Deflated Sharpe {oos['dsr_own']:.2f} (needs 0.95 to count as evidence)."
               if np.isfinite(oos.get("dsr_own", np.nan)) else "")
    if hold is not None:
        why.append(f"In the holdout it {'beat' if hold['gap_cagr_vs_spy'] > 0 else 'lagged'} SPY by "
                   f"{abs(hold['gap_cagr_vs_spy']):.1%} a year.")
    if beat_oos and strong and (hold is None or hold["gap_cagr_vs_spy"] > 0):
        label = "holds up" if hold is not None else "promising, holdout not run"
    elif beat_oos:
        label = "weak: beat SPY but could be luck"
    else:
        label = "does not hold up"
    return label, " ".join(w for w in why if w)


def _curves(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Growth of 1 for each strategy's default settings, plus SPY, same dates."""
    used = holdout_used(cfg)
    md, splits = research_view(load_market(cfg), cfg, holdout=used)
    end = splits.snapshot_end if used else splits.dev_end
    curves = {}
    for name in ORDER:
        p = params_for(cfg, name)
        run = run_window(md, cfg, build_targets(md, p), splits.data_start, end)
        if run:
            curves[name] = run["on"].returns
    first = min(r.index[0] for r in curves.values())
    ec = cfg["engine"]
    spy = run_backtest(md.open, md.close, buy_and_hold_targets(md.close, cfg["universe"]["benchmark"], first),
                       cost_rate(cfg), ec["initial_capital"], ec["cash_rate"], start=first, end=end).returns
    spy_g = (1 + spy).cumprod()
    # each line starts on its own first day at SPY's value that day, so the gap
    # to SPY always means "how it did since it started"
    growth = pd.DataFrame({k: (1 + r).cumprod() * spy_g.loc[r.index[0]] for k, r in curves.items()}).reindex(spy_g.index)
    growth["SPY"] = spy_g
    bands = [{"from": str(first.date()), "to": str(splits.insample_end.date()), "label": "In sample"},
             {"from": f"{splits.wf_first_test_year}-01-01", "to": str(splits.dev_end.date()), "label": "Walk forward test"}]
    if used:
        bands.append({"from": str(splits.holdout_start.date()), "to": str(end.date()), "label": "Holdout"})
    return growth, {"bands": bands, "holdout_used": used, "first": str(first.date()), "end": str(end.date())}


def _series_json(growth: pd.DataFrame, names: list[str]) -> dict:
    wk = growth[names].resample("W-FRI").last().dropna(how="all").astype(float)
    return {"dates": [d.strftime("%Y-%m-%d") for d in wk.index],
            "series": [{"key": n, "name": NAMES.get(n, n),
                        "values": [round(float(x), 4) if np.isfinite(x) else None for x in wk[n]]} for n in names]}


def build_report(cfg: dict) -> str:
    rd = results_dir(cfg)
    p3, p4, p5 = _read(rd / "phase3" / "summary.csv"), _read(rd / "phase4" / "summary.csv"), _read(rd / "phase5" / "summary.csv")
    picks = _read(rd / "phase4" / "walkforward_picks.csv")
    signals = _read(rd / "signals_latest.csv")
    growth, meta = _curves(cfg)
    growth.to_csv(rd / "report_curves.csv")
    trials = pd.read_csv(rd / "trials.csv") if (rd / "trials.csv").exists() else pd.DataFrame()
    n_trials = int(trials[["strategy", "params"]].drop_duplicates().shape[0]) if len(trials) else 0

    def row(df, name, col="strategy"):
        if df.empty:
            return None
        m = df[df[col] == name]
        return m.iloc[0] if len(m) else None

    # verdicts
    verdicts = {}
    for name in ORDER:
        hold = row(p5[p5["settings"] == "default"], name, "base") if not p5.empty else None
        verdicts[name] = _verdict(row(p4, name), hold)
    winners = [n for n, (lab, _) in verdicts.items() if lab == "holds up"]
    if not p4.empty and not winners:
        headline = "None of these strategies beat buying SPY once costs, multiple testing and out of sample data are accounted for."
    elif winners:
        headline = f"{len(winners)} of {len(ORDER)} held up: {', '.join(winners)}. Read the caveats before trusting it."
    else:
        headline = "Walk forward results are not in yet."

    # per strategy tables
    blocks = []
    for name in ORDER:
        periods = [("In sample", row(p3, name)), ("Walk forward (out of sample)", row(p4, name))]
        if not p5.empty:
            periods += [("Holdout, default settings", row(p5[p5["settings"] == "default"], name, "base")),
                        ("Holdout, walk forward pick", row(p5[p5["settings"] == "wf_pick"], name, "base"))]
        periods = [(lab, r) for lab, r in periods if r is not None]
        if not periods:
            continue
        head = "".join(f"<th scope='col'>{html.escape(lab)}<div class='sub'>{r['start']} to {r['end']}</div></th>" for lab, r in periods)
        body = ""
        for col, label, kind in METRICS:
            if all(col not in r.index or pd.isna(r.get(col)) for _, r in periods):
                continue
            cells = "".join(f"<td class='{('neg' if kind in ('gap', 'gapnum') and float(r.get(col, 0) or 0) < 0 else '')}'>{_fmt(r.get(col), kind)}</td>" for _, r in periods)
            body += f"<tr><th scope='row'>{label}</th>{cells}</tr>"
        lab, why = verdicts[name]
        oos = row(p4, name)
        sens = ""
        if oos is not None and "grid_sharpe_min" in oos.index:
            sens = (f"<p class='note'>Sensitivity: moving each lookback 20% either way gave a development Sharpe between "
                    f"{oos['grid_sharpe_min']:.2f} and {oos['grid_sharpe_max']:.2f} (default {oos['default_dev_sharpe']:.2f}); "
                    f"{oos['grid_share_beating_spy']:.0%} of settings beat SPY on CAGR.</p>")
        flags = [str(r.get("flags")) for _, r in periods if isinstance(r.get("flags"), str) and r.get("flags")]
        flag_html = "".join(f"<p class='flag'>⚠ Too good to trust, check for bugs: {html.escape(f)}</p>" for f in flags)
        picked = ""
        if not picks.empty:
            pk = picks[picks["strategy"] == name]
            if len(pk):
                picked = "<details><summary>Walk forward picks by year</summary><table class='mini'><tr><th>Year</th><th>Settings picked on earlier years</th><th>Return</th><th>SPY</th></tr>" + "".join(
                    f"<tr><td>{r.test_year}</td><td><code>{html.escape(str(r.picked))}</code></td><td>{_fmt(r.test_return, 'pct')}</td><td>{_fmt(r.spy_return, 'pct')}</td></tr>" for r in pk.itertuples()) + "</table></details>"
        surv = f"<p class='warn'>{SURVIVORSHIP_WARNING}</p>" if uses_single_stocks(params_for(cfg, name)) else ""
        cls = "good" if lab == "holds up" else "bad" if lab == "does not hold up" else "mid"
        blocks.append(f"""<section class='strat' id='s-{name}'>
<h3>{html.escape(NAMES[name])} <span class='verdict {cls}'>{html.escape(lab)}</span></h3>
<p>{html.escape(why)}</p>{surv}{flag_html}
<div class='tablewrap'><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>{sens}{picked}</section>""")

    # returns by year
    yr = pd.DataFrame({n: by_year(growth[n].pct_change(fill_method=None)) for n in growth.columns})
    yr_html = "<table class='years'><thead><tr><th scope='col'>Year</th>" + "".join(f"<th scope='col'>{n}</th>" for n in yr.columns) + "</tr></thead><tbody>" + "".join(
        "<tr><th scope='row'>" + str(y) + "</th>" + "".join(f"<td class='{'neg' if v < 0 else ''}'>{_fmt(v, 'pct')}</td>" for v in r) + "</tr>" for y, r in yr.iterrows()) + "</tbody></table>"

    # signals
    if signals.empty:
        sig_html = "<p>No signal list yet. It is built after the holdout run (python -m qm signals).</p>"
    else:
        sig_html = ""
        for name, g in signals.groupby("strategy", sort=False):
            items = ", ".join(f"<b>{html.escape(r.ticker)}</b> {html.escape(r.action)}" for r in g.itertuples())
            sig_html += f"<p><span class='tag'>{html.escape(name)}</span> {items}</p>"

    main = [n for n in MAIN if n in growth.columns] + ["SPY"]
    chart = _series_json(growth, main)
    chart.update(bands=meta["bands"])
    small = {v: _series_json(growth, [v, "SPY"]) for v in VARIANTS if v in growth.columns}
    table_view = growth[main].resample("YE").last()
    tv_html = "<table class='years'><thead><tr><th scope='col'>Year end</th>" + "".join(f"<th scope='col'>{n}</th>" for n in main) + "</tr></thead><tbody>" + "".join(
        f"<tr><th scope='row'>{d.year}</th>" + "".join(f"<td>{_fmt(v, 'num')}</td>" for v in r) + "</tr>" for d, r in table_view.iterrows()) + "</tbody></table>"

    banners = ""
    if cfg.get("synthetic"):
        banners += f"<div class='banner synth'><b>{html.escape(SYNTHETIC_WARNING)}</b></div>"
    banners += f"<div class='banner'>{html.escape(SURVIVORSHIP_WARNING)}</div>"
    c = cfg["costs"]
    page = TEMPLATE.format(
        banners=banners, headline=html.escape(headline), blocks="\n".join(blocks), years=yr_html,
        signals=sig_html, table_view=tv_html, n_trials=n_trials,
        costs=f"{c['commission_bps']} bps commission + {c['slippage_bps']} bps slippage per side",
        period=f"{meta['first']} to {meta['end']}",
        holdout_note="included (shaded)" if meta["holdout_used"] else "not run yet, so the chart stops before it",
        chart=json.dumps(chart), small=json.dumps(small),
        small_titles=json.dumps({k: NAMES[k] for k in small}), family=json.dumps({k: SLOT[v] for k, v in VARIANTS.items()}),
        slots=json.dumps(SLOT))
    out = rd / "report.html"
    out.write_text(page)
    return str(out)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Backtest Report</title>
<style>
:root {{ color-scheme: light;
  --surface-0:#f5f4f1; --surface-1:#fcfcfb; --border:#e3e2dd; --grid:#ecebe7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7974;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --bench:#52514e;
  --band:#0b0b0b08; --band2:#2a78d610; --band3:#eb683414;
  --good:#0ca30c; --bad:#d03b3b; --mid:#b07a00; --warnbg:#fff6e5; --synthbg:#fde8e8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
  --surface-0:#121211; --surface-1:#1a1a19; --border:#33332f; --grid:#2a2a27;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --bench:#c3c2b7;
  --band:#ffffff08; --band2:#3987e518; --band3:#d9592620; --warnbg:#2e2615; --synthbg:#3a1c1c; }} }}
:root[data-theme="dark"] {{ color-scheme: dark;
  --surface-0:#121211; --surface-1:#1a1a19; --border:#33332f; --grid:#2a2a27;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --bench:#c3c2b7;
  --band:#ffffff08; --band2:#3987e518; --band3:#d9592620; --warnbg:#2e2615; --synthbg:#3a1c1c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface-0); color:var(--text-primary);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:36px 0 10px; }} h3 {{ font-size:16px; margin:0 0 6px; }}
.lede {{ font-size:18px; margin:14px 0; }}
.meta, .note, .sub {{ color:var(--text-secondary); font-size:13px; }}
.sub {{ font-weight:400; }}
.banner {{ background:var(--warnbg); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin:10px 0; font-size:13px; }}
.banner.synth {{ background:var(--synthbg); font-size:14px; }}
.card, .strat {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; margin:14px 0; }}
.verdict {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:99px; border:1px solid currentColor; margin-left:6px; white-space:nowrap; }}
.verdict.good {{ color:var(--good); }} .verdict.bad {{ color:var(--bad); }} .verdict.mid {{ color:var(--mid); }}
.warn {{ font-size:12px; color:var(--text-secondary); border-left:3px solid var(--mid); padding-left:8px; }}
.flag {{ font-size:13px; color:var(--bad); font-weight:600; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; width:100%; }}
th, td {{ padding:5px 8px; border-bottom:1px solid var(--grid); text-align:right; }}
th[scope=row], thead th:first-child {{ text-align:left; color:var(--text-secondary); font-weight:500; }}
thead th {{ vertical-align:bottom; }}
td.neg {{ color:var(--bad); }}
table.years {{ width:auto; }} table.mini {{ width:auto; margin-top:6px; }}
code {{ font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; margin:4px 0 8px; font-size:13px; color:var(--text-secondary); }}
.legend button {{ all:unset; cursor:pointer; display:inline-flex; align-items:center; gap:6px; }}
.legend button[aria-pressed=false] {{ opacity:.35; }}
.sw {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
.chart {{ position:relative; }}
.chart svg {{ display:block; width:100%; }}
.tip {{ position:absolute; pointer-events:none; background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; font-size:12px; box-shadow:0 4px 14px #0002; display:none; min-width:180px; z-index:2; }}
.tip .r {{ display:flex; justify-content:space-between; gap:12px; }}
.grid3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
.tag {{ display:inline-block; font-weight:600; min-width:70px; }}
details summary {{ cursor:pointer; color:var(--text-secondary); font-size:13px; margin-top:8px; }}
</style></head>
<body><main>
<h1>Strategy Backtest Report</h1>
<p class="meta">Six strategies, one engine. Signals from the close of day t trade at the open of day t+1. Costs: {costs}. Trials logged: {n_trials}.</p>
{banners}
<p class="lede"><b>Bottom line:</b> {headline}</p>

<h2>Growth of $1, default settings, after costs</h2>
<p class="note">Log scale. {period}. Each strategy's line starts on its own first trading day at SPY's value that day (E and F start later because SEC XBRL data begins around 2009 to 2011). The holdout is {holdout_note}. Shaded bands mark the in sample, walk forward and holdout periods. Hover for values; click a legend item to hide it.</p>
<div class="card"><div class="legend" id="lg"></div><div class="chart" id="main"></div>
<details><summary>Table view (year end value of $1)</summary><div class="tablewrap">{table_view}</div></details></div>

<h3 style="margin-top:20px">Variants</h3>
<div class="grid3" id="small"></div>

<h2>Strategy by strategy</h2>
<p class="note">Deflated Sharpe is the chance the real Sharpe is above what the best of that many random tries would show (Bailey and Lopez de Prado). Below 0.95 it is not evidence. "Own trials" counts settings tried for that strategy; "all trials" counts everything tried in this project, which is the stricter test.</p>
{blocks}

<h2>Returns by calendar year (default settings, after costs)</h2>
<div class="card tablewrap">{years}</div>

<h2>Paper signals as of the latest close</h2>
<div class="card"><p class="note">Research only. Not advice. No orders are placed by this tool.</p>{signals}</div>

<h2>How to read this</h2>
<div class="card note">
<p>Walk forward: for each test year, settings are picked using only earlier years, then run on that year alone and stitched together. The holdout (last two years) was run once with settings fixed before it was loaded.</p>
<p>Survivorship bias cannot be removed with free data. The equal weight universe row shows how much any stock strategy gets just from owning today's survivors. Beating SPY by less than that gap is not evidence of skill.</p>
<p>Fundamentals are used from their SEC filing date, never their period end date. Missing data means the name was skipped and logged, never filled.</p>
</div>
</main>
<script>
const CH = {chart}, SMALL = {small}, STITLE = {small_titles}, FAM = {family}, SLOT = {slots};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = k => k === "SPY" ? css("--bench") : css("--s" + (SLOT[k] || FAM[k] || 1));
const fmt = v => "$" + v.toFixed(2);
function draw(el, data, opts) {{
  const W = el.clientWidth || 600, H = opts.h, m = {{l: 44, r: 12, t: 10, b: 26}};
  const n = data.dates.length, hidden = opts.hidden || new Set();
  const vis = data.series.filter(s => !hidden.has(s.key));
  let lo = Infinity, hi = -Infinity;
  vis.forEach(s => s.values.forEach(v => {{ if (v > 0) {{ lo = Math.min(lo, v); hi = Math.max(hi, v); }} }}));
  if (!isFinite(lo)) {{ lo = 0.5; hi = 2; }}
  lo = Math.log(lo * 0.95); hi = Math.log(hi * 1.05);
  const x = i => m.l + (W - m.l - m.r) * i / Math.max(n - 1, 1);
  const y = v => m.t + (H - m.t - m.b) * (1 - (Math.log(v) - lo) / (hi - lo));
  const idx = d => {{ let a = 0, b = n - 1; while (a < b) {{ const c = (a + b) >> 1; data.dates[c] < d ? a = c + 1 : b = c; }} return a; }};
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" height="${{H}}" role="img" aria-label="${{opts.label}}">`;
  (data.bands || []).forEach((b, k) => {{ const x0 = x(idx(b.from)), x1 = x(idx(b.to));
    s += `<rect x="${{x0}}" y="${{m.t}}" width="${{Math.max(x1 - x0, 0)}}" height="${{H - m.t - m.b}}" fill="var(--band${{k ? k + 1 : ''}})"/>`;
    if (x1 - x0 > b.label.length * 6.5 + 8)
      s += `<text x="${{x0 + 4}}" y="${{m.t + 12}}" font-size="11" fill="var(--text-muted)">${{b.label}}</text>`; }});
  const ticks = [];
  [0.0625, 0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 32].forEach(v => {{ const lv = Math.log(v); if (lv > lo && lv < hi) ticks.push(v); }});
  ticks.forEach(v => {{ s += `<line x1="${{m.l}}" x2="${{W - m.r}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="var(--grid)"/>`;
    s += `<text x="${{m.l - 6}}" y="${{y(v) + 4}}" font-size="11" text-anchor="end" fill="var(--text-muted)">$${{v}}</text>`; }});
  let lastYear = "", step = Math.ceil((n / 52) / Math.max(1, (W - m.l) / 70));
  data.dates.forEach((d, i) => {{ const yr = d.slice(0, 4); if (yr !== lastYear) {{ lastYear = yr;
    if ((+yr) % step === 0) s += `<text x="${{x(i)}}" y="${{H - 8}}" font-size="11" text-anchor="middle" fill="var(--text-muted)">${{yr}}</text>`; }} }});
  vis.forEach(se => {{ let p = ""; se.values.forEach((v, i) => {{ if (v > 0) p += (p ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1); }});
    s += `<path d="${{p}}" fill="none" stroke="${{color(se.key)}}" stroke-width="2" ${{se.key === "SPY" ? 'stroke-dasharray="5 4"' : ''}} stroke-linejoin="round"/>`; }});
  s += `<line id="xh" y1="${{m.t}}" y2="${{H - m.b}}" stroke="var(--text-muted)" stroke-width="1" visibility="hidden"/>`;
  s += `<rect x="${{m.l}}" y="0" width="${{W - m.l - m.r}}" height="${{H}}" fill="transparent" class="hit"/></svg><div class="tip"></div>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = el.querySelector("#xh");
  const move = ev => {{ const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("visibility", "visible");
    const rows = vis.map(se => [se, se.values[i]]).sort((a, b) => b[1] - a[1]);
    tip.innerHTML = `<b>${{data.dates[i]}}</b>` + rows.map(([se, v]) => `<div class="r"><span><span class="sw" style="background:${{color(se.key)}}"></span> ${{se.key}}</span><span>${{fmt(v)}}</span></div>`).join("");
    tip.style.display = "block"; const left = (x(i) / W) * r.width;
    tip.style.left = (left > r.width / 2 ? left - tip.offsetWidth - 12 : left + 12) + "px"; tip.style.top = "8px"; }};
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerleave", () => {{ tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); }});
}}
const hidden = new Set(), main = document.getElementById("main"), lg = document.getElementById("lg");
CH.series.forEach(se => {{ const b = document.createElement("button"); b.setAttribute("aria-pressed", "true");
  b.innerHTML = `<span class="sw" style="background:${{color(se.key)}}${{se.key === 'SPY' ? ';opacity:.8' : ''}}"></span>${{se.name}}${{se.key === 'SPY' ? ' (dashed)' : ''}}`;
  b.onclick = () => {{ hidden.has(se.key) ? hidden.delete(se.key) : hidden.add(se.key); b.setAttribute("aria-pressed", String(!hidden.has(se.key))); render(); }};
  lg.appendChild(b); }});
const smallEl = document.getElementById("small");
Object.keys(SMALL).forEach(k => {{ const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<h3>${{STITLE[k]}}</h3><div class="legend"><span><span class="sw" style="background:${{color(k)}}"></span>${{k}}</span><span><span class="sw" style="background:${{color('SPY')}}"></span>SPY (dashed)</span></div><div class="chart"></div>`;
  smallEl.appendChild(c); }});
function render() {{
  draw(main, CH, {{h: 380, hidden, label: "Growth of one dollar for each strategy and SPY"}});
  smallEl.querySelectorAll(".chart").forEach((el, i) => {{ const k = Object.keys(SMALL)[i]; draw(el, SMALL[k], {{h: 200, label: k + " versus SPY"}}); }});
}}
render(); addEventListener("resize", render);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script></body></html>
"""
