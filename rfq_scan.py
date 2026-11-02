"""Request market-maker quotes on every 3-leg Senate combo and price each with the engine.

Dry run by default: reads public market data, enumerates combos, prices them,
and writes the plan. Nothing is created and no RFQ is sent without --send.

With --send, for each wave of up to 100 combos (Kalshi's open-RFQ limit):
  1. create the combo market (POST /multivariate_event_collections/{collection});
     tickers are cached in data/rfq/combos.csv so a combo is never created twice
     (Kalshi allows 5,000 creations per week)
  2. open an RFQ on it (rest_remainder=false, so nothing ever rests on the book)
  3. wait, then read every quote
  4. delete the RFQ
Quotes are never accepted. Open RFQs are deleted even if the run fails.

Correlation model: one national factor. Every non-Republican leg (D or an
independent challenger) loads positively on it and every Republican leg
negatively, so a D win in one state and an R win in another are negatively
correlated. --rho is the common latent correlation; the cross-state Senate rho
is not calibrated anywhere in this repo, so the default is a placeholder.

    python rfq_scan.py                         # dry run, writes data/rfq/plan.csv
    python rfq_scan.py --send --max-combos 100 # one live wave
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pricing import combo_prob

PUBLIC_BASE = "https://external-api.kalshi.com/trade-api/v2"
COLLECTION = "KXMVECROSSCATEGORY-R"
EVENT_PREFIX = "SENATE"
LEGS = 3
MAX_OPEN_RFQS = 100
WEEKLY_CREATION_LIMIT = 5000
OUT_DIR = Path(__file__).parent / "data" / "rfq"


@dataclass(frozen=True)
class Leg:
    event_ticker: str
    market_ticker: str
    label: str
    prob: float  # mid of the leg's own market

    @property
    def is_republican(self) -> bool:
        return self.market_ticker.rsplit("-", 1)[-1] == "R"


def engine_price(legs: tuple[Leg, ...], rho: float) -> float:
    """P(all legs win). Republican legs are the NO side of an anti-R latent."""
    probs = [1.0 - l.prob if l.is_republican else l.prob for l in legs]
    sides = ["no" if l.is_republican else "yes" for l in legs]
    return combo_prob(probs, sides, rho)


def implied_rho(legs: tuple[Leg, ...], price: float) -> float:
    """rho at which the engine reproduces price, or NaN if none in [0, 0.999] does."""
    f = lambda r: engine_price(legs, r) - price
    if not np.isfinite(price) or np.sign(f(0.0)) == np.sign(f(0.999)):
        return np.nan
    return brentq(f, 0.0, 0.999, xtol=1e-8)


def _public_get(path: str) -> dict:
    with urllib.request.urlopen(PUBLIC_BASE + path, timeout=20) as resp:
        return json.load(resp)


def load_races(collection: str = COLLECTION, prefix: str = EVENT_PREFIX) -> dict[str, list[Leg]]:
    """Comboable races in the collection -> their active markets as legs."""
    coll = _public_get(f"/multivariate_event_collections/{collection}")["multivariate_contract"]
    races = {}
    for ev in sorted(e for e in coll["associated_event_tickers"] if e.startswith(prefix)):
        markets = _public_get(f"/events/{ev}?with_nested_markets=true")["event"]["markets"]
        legs = []
        for m in markets:
            if m.get("status") not in ("active", "open"):
                continue
            bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
            legs.append(Leg(ev, m["ticker"], m.get("yes_sub_title") or m["ticker"], (bid + ask) / 2))
        if legs:
            races[ev] = legs
    return races


def enumerate_combos(races: dict[str, list[Leg]], n_legs: int = LEGS) -> list[tuple[Leg, ...]]:
    """One market from each of n_legs different races (all-YES combos)."""
    combos = []
    for events in itertools.combinations(sorted(races), n_legs):
        combos.extend(itertools.product(*(races[e] for e in events)))
    return combos


def combo_key(legs: tuple[Leg, ...]) -> str:
    return "|".join(sorted(l.market_ticker for l in legs))


def _read_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["key"]: r["combo_ticker"] for r in csv.DictReader(f)}


def _append_cache(path: Path, key: str, ticker: str) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "combo_ticker", "created_at"])
        w.writerow([key, ticker, datetime.now(timezone.utc).isoformat()])


def run_wave(client, wave: list[tuple[Leg, ...]], cache: dict[str, str], cache_path: Path,
             contracts: float, wait: float, collection: str = COLLECTION,
             obscure_id: bool = True, log=print) -> list[dict]:
    """Create combos, open RFQs, collect quotes, delete RFQs. Never accepts a quote."""
    opened: list[tuple[tuple[Leg, ...], str, str]] = []  # (legs, combo_ticker, rfq_id)
    results = []
    try:
        for legs in wave:
            key = combo_key(legs)
            ticker = cache.get(key)
            if ticker is None:
                body = {"selected_markets": [
                    {"market_ticker": l.market_ticker, "event_ticker": l.event_ticker, "side": "yes"}
                    for l in legs]}
                ticker = client.request("POST", f"/multivariate_event_collections/{collection}",
                                        body=body)["market_ticker"]
                cache[key] = ticker
                _append_cache(cache_path, key, ticker)
            rfq = client.request("POST", "/communications/rfqs", body={
                "market_ticker": ticker,
                "contracts_fp": f"{contracts:.2f}",
                "rest_remainder": False,
                "obscure_creator_id": obscure_id,
            })
            opened.append((legs, ticker, rfq["id"]))
        log(f"  {len(opened)} RFQs open; waiting {wait:.0f}s for quotes")
        time.sleep(wait)
        for legs, ticker, rfq_id in opened:
            # Kalshi rejects a quote query unless it names the RFQ creator (403)
            quotes = client.get("/communications/quotes", rfq_id=rfq_id,
                                rfq_user_filter="self").get("quotes", [])
            yes = [float(q["yes_bid_dollars"]) for q in quotes if float(q.get("yes_bid_dollars") or 0) > 0]
            no = [float(q["no_bid_dollars"]) for q in quotes if float(q.get("no_bid_dollars") or 0) > 0]
            results.append({"legs": legs, "combo_ticker": ticker, "rfq_id": rfq_id,
                            "n_quotes": len(quotes),
                            "best_yes_bid": max(yes) if yes else np.nan,
                            "best_no_bid": max(no) if no else np.nan,
                            "quoted_at": datetime.now(timezone.utc).isoformat()})
    finally:
        for _, _, rfq_id in opened:
            try:
                client.request("DELETE", f"/communications/rfqs/{rfq_id}")
            except Exception as exc:  # keep deleting the rest
                log(f"  could not delete RFQ {rfq_id}: {exc}")
    return results


def result_row(r: dict, rho: float) -> dict:
    legs = r["legs"]
    engine = engine_price(legs, rho)
    yes_bid, no_bid = r["best_yes_bid"], r["best_no_bid"]
    yes_ask = 1.0 - no_bid  # buying YES = selling NO to the maker
    mid = (yes_bid + yes_ask) / 2 if np.isfinite(yes_bid) and np.isfinite(no_bid) else np.nan
    return {
        "combo_ticker": r["combo_ticker"],
        "legs": " & ".join(l.label for l in legs),
        "leg_markets": combo_key(legs),
        "leg_probs": " ".join(f"{l.prob:.4f}" for l in legs),
        "rho": rho,
        "engine": engine,
        "n_quotes": r["n_quotes"],
        "maker_yes_bid": yes_bid,
        "maker_yes_ask": yes_ask,
        "maker_mid": mid,
        "engine_minus_mid": engine - mid,
        "maker_implied_rho": implied_rho(legs, mid),
        "rfq_id": r["rfq_id"],
        "quoted_at": r["quoted_at"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rho", type=float, default=0.3, help="engine rho (placeholder default)")
    ap.add_argument("--send", action="store_true", help="actually create combos and send RFQs")
    ap.add_argument("--env", default="prod", choices=["prod", "demo"])
    ap.add_argument("--max-combos", type=int, default=100, help="stop after this many combos")
    ap.add_argument("--contracts", type=float, default=100.0, help="RFQ size in contracts")
    ap.add_argument("--wait", type=float, default=15.0, help="seconds to collect quotes per wave")
    ap.add_argument("--show-id", action="store_true", help="don't hide your requester ID from makers")
    args = ap.parse_args()

    races = load_races()
    combos = enumerate_combos(races)
    # Most likely combos first: with the weekly creation limit below the full
    # universe, the near-zero combos (least informative on a 0.1c grid) go last.
    combos.sort(key=lambda legs: -engine_price(legs, args.rho))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "combos.csv"
    cache = _read_cache(cache_path)
    todo = combos[: args.max_combos]
    new_creations = sum(combo_key(c) not in cache for c in todo)

    print(f"{len(races)} Senate races, {len(combos)} {LEGS}-leg combos; rho = {args.rho} "
          f"(placeholder unless you set it)")
    print(f"this run: {len(todo)} combos, {new_creations} new combo creations "
          f"({len(cache)} already created; weekly limit {WEEKLY_CREATION_LIMIT})")

    if not args.send:
        plan = OUT_DIR / "plan.csv"
        plan.unlink(missing_ok=True)
        _write_csv(plan, [{"rank": i + 1, "legs": " & ".join(l.label for l in c),
                           "leg_markets": combo_key(c), "engine": engine_price(c, args.rho)}
                          for i, c in enumerate(combos)])
        print(f"dry run: nothing created or sent. Full ranked plan in {plan}")
        for i, c in enumerate(combos[:5]):
            print(f"  {i + 1}. {engine_price(c, args.rho):.4f}  {' & '.join(l.label for l in c)}")
        return

    from kalshi_client import KalshiClient
    client = KalshiClient(args.env)
    out = OUT_DIR / f"quotes_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    for start in range(0, len(todo), MAX_OPEN_RFQS):
        wave = todo[start : start + MAX_OPEN_RFQS]
        print(f"wave {start // MAX_OPEN_RFQS + 1}: combos {start + 1}-{start + len(wave)}")
        rows = [result_row(r, args.rho)
                for r in run_wave(client, wave, cache, cache_path, args.contracts, args.wait,
                                  obscure_id=not args.show_id)]
        _write_csv(out, rows)
        quoted = sum(r["n_quotes"] > 0 for r in rows)
        print(f"  {quoted}/{len(rows)} combos got quotes -> {out.name}")


if __name__ == "__main__":
    main()
