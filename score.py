"""Score the engine against market makers once the Senate races resolve.

Every RFQ scan file (data/rfq/quotes_*.csv) already holds, per combo, the
engine price at the rho chosen before the election and the makers' best bid
and ask. Those files are committed and tagged before polls close, so the only
new input here is the outcomes.

Outcomes come from each leg market's Kalshi result. Races called by the media
before Kalshi settles can be added by hand in a CSV (market_ticker,result with
result yes/no), which overrides the API. A combo resolves NO as soon as any leg
does, and YES once every leg has.

Scores, on combos with a two-sided maker quote and a known outcome:
  Brier and log loss for the engine, the maker mid, and independence (rho = 0)

The combos share ~18 races and one national swing, so they are not independent
trials: one election cannot establish skill however many combos score.

    python score.py                         # all scan files, outcomes from Kalshi
    python score.py --results called.csv    # plus hand-entered race calls
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC_BASE = "https://external-api.kalshi.com/trade-api/v2"
RFQ_DIR = Path(__file__).parent / "data" / "rfq"
LOG_EPS = 1e-6


def fetch_results(tickers, timeout: float = 20) -> dict[str, str]:
    """market ticker -> 'yes' / 'no' for markets Kalshi has resolved."""
    out = {}
    for t in sorted(set(tickers)):
        try:
            with urllib.request.urlopen(f"{PUBLIC_BASE}/markets/{t}", timeout=timeout) as resp:
                result = json.load(resp)["market"].get("result")
        except urllib.error.HTTPError:
            continue
        if result in ("yes", "no"):
            out[t] = result
    return out


def read_results(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = {r["market_ticker"].strip(): r["result"].strip().lower() for r in csv.DictReader(f)}
    bad = {t: r for t, r in rows.items() if r not in ("yes", "no")}
    if bad:
        raise ValueError(f"results must be yes/no: {bad}")
    return rows


def combo_outcome(leg_markets: str, results: dict[str, str]) -> float:
    """1 if every leg resolved yes, 0 if any resolved no, NaN while undecided."""
    legs = [results.get(t) for t in leg_markets.split("|")]
    if "no" in legs:
        return 0.0
    if all(r == "yes" for r in legs):
        return 1.0
    return np.nan


def _brier(p, y) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p, y) -> float:
    p = np.clip(p, LOG_EPS, 1 - LOG_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(df: pd.DataFrame, results: dict[str, str]) -> dict:
    df = df.assign(outcome=[combo_outcome(m, results) for m in df.leg_markets])
    df["independent"] = df.leg_probs.str.split().apply(lambda v: float(np.prod([float(x) for x in v])))
    known = df[df.outcome.notna()]
    two = known[known.maker_mid.notna()]
    out = {"rows": len(df), "resolved": len(known), "two_sided_resolved": len(two),
           "rho": float(df.rho.iloc[0]) if len(df) else np.nan}
    y = two.outcome.to_numpy()
    for name in ("engine", "maker_mid", "independent"):
        p = two[name].to_numpy()
        out[f"brier_{name}"] = _brier(p, y) if len(two) else np.nan
        out[f"logloss_{name}"] = _log_loss(p, y) if len(two) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path, help="scan files (default: all in data/rfq)")
    ap.add_argument("--results", type=Path, help="CSV market_ticker,result overriding Kalshi")
    args = ap.parse_args()

    files = args.files or sorted(RFQ_DIR.glob("quotes_*.csv"))
    frames = {f: pd.read_csv(f) for f in files}
    tickers = {t for df in frames.values() for m in df.leg_markets for t in m.split("|")}
    results = fetch_results(tickers)
    if args.results:
        results.update(read_results(args.results))
    print(f"{len(results)} of {len(tickers)} leg markets resolved "
          f"({sum(r == 'yes' for r in results.values())} yes)")

    for f, df in frames.items():
        s = score(df, results)
        print(f"\n{f.name}: rho {s['rho']:g}, {s['resolved']}/{s['rows']} combos resolved, "
              f"{s['two_sided_resolved']} with a two-sided quote")
        if s["two_sided_resolved"]:
            print("               brier   log loss   (lower is better)")
            for name in ("engine", "maker_mid", "independent"):
                print(f"  {name:<12} {s['brier_' + name]:.4f}   {s['logloss_' + name]:.4f}")


if __name__ == "__main__":
    main()
