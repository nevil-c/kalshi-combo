"""Compare one snapshot's Kalshi combo prices with the engine's prices.

The engine is fed that snapshot's own House and Senate control prices, so both
sides see the same underlying legs. rho is either given (--rho) or fitted to
the same snapshot, in which case the comparison is in-sample.

    python compare.py                                  # latest snapshot, fitted rho
    python compare.py data/snapshot_X.json --rho 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from calibrate import CELLS, COMBO_EVENT, HOUSE, SENATE, calibrate_snapshot, leg_prob
from parse import load_snapshot
from pricing import combo_prob


def compare(quotes: pd.DataFrame, rho: float | None = None,
            event: str = COMBO_EVENT) -> tuple[pd.DataFrame, dict]:
    """Per combo cell: Kalshi bid/ask/mid, engine price, and where it falls."""
    q = quotes.set_index("ticker")
    p_h = leg_prob(q["mid"], *HOUSE)
    p_s = leg_prob(q["mid"], *SENATE)
    fitted = rho is None
    if fitted:
        rho = calibrate_snapshot(quotes, event).rho

    rows = []
    for cell, sides in CELLS.items():
        ticker = f"{event}-{cell}"
        if ticker not in q.index:
            continue
        bid, ask, mid = q.loc[ticker, ["bid", "ask", "mid"]]
        engine = combo_prob([p_h, p_s], sides, rho)
        if engine < bid:
            where = "below bid"
        elif engine > ask:
            where = "above ask"
        else:
            where = "inside"
        rows.append({"cell": cell, "bid": bid, "ask": ask, "mid": mid,
                     "engine": engine, "engine_minus_mid": engine - mid, "vs_quote": where})
    info = {"p_house_d": p_h, "p_senate_d": p_s, "rho": rho, "rho_fitted": fitted}
    return pd.DataFrame(rows).set_index("cell"), info


def _latest_snapshot(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("snapshot_*.json"))
    if not files:
        raise SystemExit(f"no snapshots in {data_dir}; run collector.py first")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot", nargs="?", type=Path, help="snapshot file (default: latest in data/)")
    ap.add_argument("--rho", type=float, help="engine rho (default: fit to this snapshot)")
    args = ap.parse_args()

    path = args.snapshot or _latest_snapshot(Path(__file__).parent / "data")
    quotes = load_snapshot(path)
    table, info = compare(quotes, args.rho)

    q = quotes.set_index("ticker")
    print(f"snapshot {path.name}  ({quotes['time'].iloc[0]})\n")
    print("legs (bid / ask):")
    for t in (*HOUSE, *SENATE):
        print(f"  {t:18s} {q.loc[t, 'bid']:.4f} / {q.loc[t, 'ask']:.4f}")
    print(f"  -> P(House D) = {info['p_house_d']:.4f}   P(Senate D) = {info['p_senate_d']:.4f}")
    how = "fitted to this snapshot (in-sample)" if info["rho_fitted"] else "given"
    print(f"\nrho = {info['rho']:.4f}  [{how}]\n")
    print(table.to_string(formatters={
        **{c: "{:.4f}".format for c in ("bid", "ask", "mid", "engine")},
        "engine_minus_mid": "{:+.4f}".format,
    }))
    print(f"\nengine cells sum to {table['engine'].sum():.4f}; "
          f"Kalshi mids sum to {table['mid'].sum():.4f}")


if __name__ == "__main__":
    main()
