"""Turn raw collector snapshots into a tidy quotes DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["ticker", "time", "bid", "ask", "mid"]


def _dollars(value) -> float:
    """Kalshi dollar string ("0.6100") -> float; missing/empty -> NaN."""
    if value is None or value == "":
        return np.nan
    return float(value)


def load_snapshots(data_dir: str | Path = "data") -> pd.DataFrame:
    """One row per (ticker, snapshot time): ticker, time (UTC), bid, ask, mid."""
    rows = []
    for path in sorted(Path(data_dir).glob("snapshot_*.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        for m in snap["markets"]:
            rows.append({
                "ticker": m["ticker"],
                "time": snap["fetched_at"],
                "bid": _dollars(m.get("yes_bid_dollars")),
                "ask": _dollars(m.get("yes_ask_dollars")),
            })
    df = pd.DataFrame(rows, columns=["ticker", "time", "bid", "ask"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    # NaN propagates, so mid is NaN whenever either side is missing.
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.drop_duplicates(["ticker", "time"], keep="last")
    return df.sort_values(["time", "ticker"]).reset_index(drop=True)[COLUMNS]


def latest(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent quote for each ticker."""
    return df.sort_values("time").groupby("ticker").tail(1).reset_index(drop=True)
