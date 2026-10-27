"""Save snapshots of the Kalshi markets used by the combo pricer.

Read-only: unauthenticated GET requests for public market data only.
Never places orders or RFQs.

    python collector.py                        # one snapshot
    python collector.py --loop --interval 3600 # hourly until stopped
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
QUERIES = [
    {"series_ticker": "KXBALANCEPOWERCOMBO"},
    {"event_ticker": "CONTROLH-2026"},
    {"event_ticker": "CONTROLS-2026"},
]


def _get(path: str, params: dict, timeout: float = 20.0, retries: int = 1) -> dict:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2.0)
    raise AssertionError("unreachable")


def fetch_markets(params: dict) -> list[dict]:
    markets, cursor = [], None
    while True:
        page = _get("/markets", {**params, "limit": 200, **({"cursor": cursor} if cursor else {})})
        markets.extend(page.get("markets", []))
        cursor = page.get("cursor")
        if not cursor:
            return markets


def snapshot(data_dir: Path) -> Path:
    fetched_at = datetime.now(timezone.utc)
    markets = [m for q in QUERIES for m in fetch_markets(q)]
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"snapshot_{fetched_at:%Y%m%dT%H%M%SZ}.json"
    payload = {"fetched_at": fetched_at.isoformat(), "markets": markets}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--loop", action="store_true", help="keep collecting")
    ap.add_argument("--interval", type=float, default=3600.0, help="seconds between snapshots")
    args = ap.parse_args()
    while True:
        try:
            path = snapshot(args.data)
            print(f"saved {path}")
        except OSError as exc:
            if not args.loop:
                raise
            print(f"snapshot failed: {exc}")
        if not args.loop:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
