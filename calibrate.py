"""Back out the copula rho implied by Kalshi's House x Senate combo markets.

The KXBALANCEPOWERCOMBO event has four cells (DD, DR, RD, RR; House first,
Senate second). Given the separate House and Senate control marginals, a
single rho is fitted by least squares across all four cells, and each cell
is also inverted on its own as a consistency check.

    python calibrate.py [--data data] [--weighted]
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from parse import latest, load_snapshots
from pricing import combo_prob

COMBO_EVENT = "KXBALANCEPOWERCOMBO-27FEB"
HOUSE = ("CONTROLH-2026-D", "CONTROLH-2026-R")
SENATE = ("CONTROLS-2026-D", "CONTROLS-2026-R")
# Sides on the (House D, Senate D) legs for each combo cell.
CELLS = {
    "DD": ("yes", "yes"),
    "DR": ("yes", "no"),
    "RD": ("no", "yes"),
    "RR": ("no", "no"),
}
RHO_BOUNDS = (0.0, 0.999)


@dataclass
class CalibResult:
    rho: float
    p_house_d: float
    p_senate_d: float
    overround: float  # sum of the four cell mids minus 1
    table: pd.DataFrame  # per cell: market, model, resid, indep, cell_rho


def leg_prob(quotes: Mapping[str, float], d_ticker: str, r_ticker: str) -> float:
    """P(Democratic) from both sides of the book: mean of mid(D) and 1 - mid(R)."""
    return 0.5 * (quotes[d_ticker] + 1.0 - quotes[r_ticker])


def _cell_rho(p_h: float, p_s: float, sides, target: float) -> float:
    """rho that reproduces one cell exactly, or NaN if no rho in bounds does."""
    f = lambda r: combo_prob([p_h, p_s], sides, r) - target
    lo, hi = RHO_BOUNDS
    if np.sign(f(lo)) == np.sign(f(hi)):
        return np.nan
    return brentq(f, lo, hi, xtol=1e-10)


def implied_rho(
    p_house_d: float,
    p_senate_d: float,
    cells: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> CalibResult:
    """Fit rho to the combo cell prices given the two marginals."""
    keys = [k for k in CELLS if k in cells]
    if not keys:
        raise ValueError("no combo cells to fit")
    w = {k: (weights or {}).get(k, 1.0) for k in keys}

    def sse(rho: float) -> float:
        return sum(
            w[k] * (combo_prob([p_house_d, p_senate_d], CELLS[k], rho) - cells[k]) ** 2
            for k in keys
        )

    rho = float(minimize_scalar(sse, bounds=RHO_BOUNDS, method="bounded",
                                options={"xatol": 1e-8}).x)
    rows = []
    for k in keys:
        model = combo_prob([p_house_d, p_senate_d], CELLS[k], rho)
        rows.append({
            "cell": k,
            "market": cells[k],
            "model": model,
            "resid": cells[k] - model,
            "indep": combo_prob([p_house_d, p_senate_d], CELLS[k], 0.0),
            "cell_rho": _cell_rho(p_house_d, p_senate_d, CELLS[k], cells[k]),
        })
    return CalibResult(
        rho=rho,
        p_house_d=p_house_d,
        p_senate_d=p_senate_d,
        overround=sum(cells[k] for k in keys) - 1.0,
        table=pd.DataFrame(rows).set_index("cell"),
    )


def calibrate_snapshot(quotes: pd.DataFrame, event: str = COMBO_EVENT,
                       weighted: bool = False) -> CalibResult:
    """Calibrate from one snapshot's rows of the parsed DataFrame."""
    mid = quotes.set_index("ticker")["mid"]
    spread = quotes.set_index("ticker").eval("ask - bid")
    cells = {k: mid[f"{event}-{k}"] for k in CELLS if f"{event}-{k}" in mid.index}
    cells = {k: v for k, v in cells.items() if np.isfinite(v)}
    weights = None
    if weighted:
        weights = {k: 1.0 / max(spread[f"{event}-{k}"], 0.001) for k in cells}
    return implied_rho(leg_prob(mid, *HOUSE), leg_prob(mid, *SENATE), cells, weights)


def calibrate_series(df: pd.DataFrame, event: str = COMBO_EVENT,
                     weighted: bool = False) -> pd.DataFrame:
    """rho and marginals at every snapshot time."""
    rows = []
    for t, snap in df.groupby("time"):
        try:
            res = calibrate_snapshot(snap, event, weighted)
        except (KeyError, ValueError):
            continue  # snapshot missing a leg or all cells
        rows.append({
            "time": t,
            "rho": res.rho,
            "p_house_d": res.p_house_d,
            "p_senate_d": res.p_senate_d,
            "overround": res.overround,
            "max_abs_resid": res.table["resid"].abs().max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--weighted", action="store_true", help="weight cells by 1/spread")
    args = ap.parse_args()

    df = load_snapshots(args.data)
    if df.empty:
        raise SystemExit(f"no snapshots in {args.data}; run collector.py first")
    last = latest(df)
    res = calibrate_snapshot(last, weighted=args.weighted)
    print(f"as of {last['time'].max()}")
    print(f"P(House D) = {res.p_house_d:.4f}   P(Senate D) = {res.p_senate_d:.4f}")
    print(f"combo overround = {res.overround:+.4f}")
    print(f"implied rho = {res.rho:.4f}\n")
    print(res.table.to_string(float_format=lambda x: f"{x:.4f}"))

    series = calibrate_series(df, weighted=args.weighted)
    if len(series) > 1:
        print(f"\nrho over {len(series)} snapshots: "
              f"min {series.rho.min():.4f}  median {series.rho.median():.4f}  "
              f"max {series.rho.max():.4f}")


if __name__ == "__main__":
    main()
