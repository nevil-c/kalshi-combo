"""Cross-state Senate rho implied by the Senate control market.

Senate control is a function of the individual races, so given every race's
price and a rho the copula implies a distribution for the number of
Democratic-caucus seats, and from that P(D controls the Senate). The rho at
which that matches CONTROLS-2026 is the market's cross-state Senate rho, read
off a ~1c-wide market instead of ~35c-wide combo RFQ quotes.

Same latent as rfq_scan.py: one national factor, every non-Republican outcome
loads positively and Republican outcomes negatively. Conditional on the factor,
a race goes non-Republican with the copula probability and then splits between
the Democrat and any independent in proportion to their prices.

Independents (Osborn in NE, plus long shots in MT, SD, ID) caucus with nobody
by default; --ind-weight 1 counts them as Democratic-caucus seats. Control is
decided by the President pro tempore, so with a Republican VP Democrats need
51 seats.

    python senate_rho.py                   # fit rho to the control bid/mid/ask
    python senate_rho.py --ind-weight 1    # independents caucus with Democrats
    python senate_rho.py --swing MI,TX,OH,ME,IA,AK,KS   # different swing group

Two tiers: the swing races share one pairwise rho and every other pair another
(lower) one. For each swing rho the CLI solves for the other rho that still
matches the control price; "none" means no such rho exists, so control rules
out that much swing correlation.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pricing import GH_RHO_MAX, _nodes, _step_nodes, cond_leg_prob

PUBLIC_BASE = "https://external-api.kalshi.com/trade-api/v2"
CONTROL = ("CONTROLS-2026-D", "CONTROLS-2026-R")
# The 35 seats up in 2026: 33 class-2 seats plus the FL and OH specials.
STATES = ("AK AL AR CO DE FLS GA IA ID IL KS KY LA MA ME MI MN MS MT NC NE NH NJ NM "
          "OHS OK OR RI SC SD TN TX VA WV WY").split()
NO_MARKET = {"SENATEKY-26": "R"}  # races Kalshi lists no market for, as safe seats
HOLDOVER_D = 34  # Democratic-caucus seats not up in 2026 (incl. King and Sanders)
NEEDED_D = 51  # a 50-50 tie goes to the Republican VP
RHO_GRID = np.linspace(0.0, 0.995, 200)


@dataclass(frozen=True)
class Race:
    event_ticker: str
    p_r: float  # P(Republican wins)
    p_d: float  # P(Democrat wins)
    p_ind: float = 0.0  # P(an independent wins)

    def d_seat_share(self, ind_weight: float) -> float:
        """P(Democratic-caucus seat | a non-Republican wins)."""
        non_r = self.p_d + self.p_ind
        return (self.p_d + ind_weight * self.p_ind) / non_r if non_r > 0 else 0.0


def _public_get(path: str) -> dict:
    with urllib.request.urlopen(PUBLIC_BASE + path, timeout=20) as resp:
        return json.load(resp)


def _mid(m: dict) -> float:
    return (float(m["yes_bid_dollars"]) + float(m["yes_ask_dollars"])) / 2


def race_from_markets(event_ticker: str, markets: list[dict]) -> Race:
    """Race probabilities from its markets' mids, normalized to sum to 1."""
    mids = {m["ticker"].rsplit("-", 1)[-1]: _mid(m) for m in markets
            if m.get("status") in ("active", "open")}
    total = sum(mids.values())
    p_r, p_d = mids.get("R", 0.0) / total, mids.get("D", 0.0) / total
    return Race(event_ticker, p_r, p_d, max(0.0, 1.0 - p_r - p_d))


def load_races(states=STATES) -> list[Race]:
    races = []
    for st in states:
        ev = f"SENATE{st}-26"
        if ev in NO_MARKET:
            winner = NO_MARKET[ev]
            races.append(Race(ev, float(winner == "R"), float(winner == "D")))
            continue
        markets = _public_get(f"/events/{ev}?with_nested_markets=true")["event"]["markets"]
        races.append(race_from_markets(ev, markets))
    return races


def load_control() -> tuple[float, float, float]:
    """(bid, mid, ask) for P(Democrats control), using both sides of the book."""
    m = {mk["ticker"]: mk for mk in
         _public_get("/markets?event_ticker=CONTROLS-2026")["markets"]}
    d, r = m[CONTROL[0]], m[CONTROL[1]]
    bid = max(float(d["yes_bid_dollars"]), 1.0 - float(r["yes_ask_dollars"]))
    ask = min(float(d["yes_ask_dollars"]), 1.0 - float(r["yes_bid_dollars"]))
    return bid, (_mid(d) + 1.0 - _mid(r)) / 2, ask


def _quadrature(races: list[Race], rho: float) -> tuple[np.ndarray, np.ndarray]:
    if rho == 0.0:
        return np.zeros(1), np.ones(1)
    if rho <= GH_RHO_MAX:
        return _nodes(64)
    return _step_nodes(np.array([1.0 - r.p_r for r in races]), rho)


def _cond_seat_dist(races: list[Race], rho: float, m: np.ndarray, ind_weight: float) -> np.ndarray:
    """P(k Democratic-caucus seats | factor = m), shape (len(m), len(races) + 1)."""
    dist = np.zeros((len(m), len(races) + 1))
    dist[:, 0] = 1.0
    for race in races:
        q = race.d_seat_share(ind_weight) * cond_leg_prob(1.0 - race.p_r, rho, m)
        new = dist * (1.0 - q)[:, None]
        new[:, 1:] += dist[:, :-1] * q[:, None]
        dist = new
    return dist


def seat_distribution(races: list[Race], rho: float, ind_weight: float = 0.0) -> np.ndarray:
    """P(Democratic-caucus seats won among these races = k), k = 0..len(races)."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    m, w = _quadrature(races, rho)
    return w @ _cond_seat_dist(races, rho, m, ind_weight)


def control_prob(races: list[Race], rho: float, ind_weight: float = 0.0,
                 holdover_d: int = HOLDOVER_D, needed: int = NEEDED_D) -> float:
    dist = seat_distribution(races, rho, ind_weight)
    k = max(needed - holdover_d, 0)
    return float(np.clip(dist[k:].sum(), 0.0, 1.0))


def control_prob_two_tier(races: list[Race], swing: set[str], rho_swing: float,
                          rho_base: float, ind_weight: float = 0.0,
                          holdover_d: int = HOLDOVER_D, needed: int = NEEDED_D) -> float:
    """P(control) when swing races share pairwise rho_swing and every other pair rho_base.

    Swing races load on a group factor G and the rest on the national factor M,
    with corr(G, M) = sqrt(rho_base / rho_swing). Integrates over G, then M | G.
    """
    if not 0.0 <= rho_base <= rho_swing < 1.0:
        raise ValueError("need 0 <= rho_base <= rho_swing < 1")
    if rho_base > GH_RHO_MAX:
        raise ValueError(f"rho_base above {GH_RHO_MAX} is not supported")
    sw = [r for r in races if r.event_ticker in swing]
    rest = [r for r in races if r.event_ticker not in swing]
    if rho_swing == 0.0:
        return control_prob(races, 0.0, ind_weight, holdover_d, needed)
    g, wg = _quadrature(sw, rho_swing)
    d_sw = _cond_seat_dist(sw, rho_swing, g, ind_weight)
    c = np.sqrt(rho_base / rho_swing)
    x, wx = _nodes(32)
    m = c * g[:, None] + np.sqrt(1.0 - c * c) * x[None, :]
    d_rest = _cond_seat_dist(rest, rho_base, m.ravel(), ind_weight)
    d_rest = np.einsum("gxk,x->gk", d_rest.reshape(len(g), len(x), -1), wx)
    # tail[g, j] = P(rest wins >= j | G = g), with j >= len(rest) + 1 impossible
    tail = np.concatenate([d_rest[:, ::-1].cumsum(axis=1)[:, ::-1],
                           np.zeros((len(g), len(sw) + 1))], axis=1)
    need = max(needed - holdover_d, 0)
    j = np.clip(need - np.arange(len(sw) + 1), 0, None)
    return float(np.clip(wg @ (d_sw * tail[:, j]).sum(axis=1), 0.0, 1.0))


def implied_base_rho(races: list[Race], swing: set[str], rho_swing: float, target: float,
                     **kw) -> float:
    """rho_base in [0, min(rho_swing, 0.8)] matching target, or NaN if none does."""
    hi = min(rho_swing, GH_RHO_MAX)
    f = lambda r: control_prob_two_tier(races, swing, rho_swing, r, **kw) - target
    grid = np.linspace(0.0, hi, 41)
    vals = np.array([f(r) for r in grid])
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            return float(a)
        if fa * fb < 0:
            return brentq(f, a, b, xtol=1e-5)
    return float(grid[-1]) if vals[-1] == 0.0 else np.nan


def implied_rhos(races: list[Race], target: float, **kw) -> list[float]:
    """Every rho in [0, 0.995] at which control_prob equals target (it need not be monotone)."""
    f = np.array([control_prob(races, r, **kw) for r in RHO_GRID]) - target
    roots = []
    for a, b, fa, fb in zip(RHO_GRID[:-1], RHO_GRID[1:], f[:-1], f[1:]):
        if fa == 0.0:
            roots.append(float(a))
        elif fa * fb < 0:
            roots.append(brentq(lambda r: control_prob(races, r, **kw) - target, a, b, xtol=1e-6))
    return roots


def _fmt(roots: list[float]) -> str:
    return ", ".join(f"{r:.3f}" for r in roots) if roots else "none in [0, 0.995]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ind-weight", type=float, default=None,
                    help="share of independent wins counted as Democratic caucus "
                         "(default: show 0 and 1)")
    ap.add_argument("--holdover-d", type=int, default=HOLDOVER_D)
    ap.add_argument("--swing", default="MI,TX,OH,ME",
                    help="comma-separated states whose pairs get a separate, higher rho")
    args = ap.parse_args()

    races = load_races()
    bid, mid, ask = load_control()
    need = NEEDED_D - args.holdover_d
    print(f"{len(races)} races; Democrats need {need} of them "
          f"({args.holdover_d} holdovers, {NEEDED_D} for control)")
    print(f"Senate control (D): bid {bid:.3f}  mid {mid:.3f}  ask {ask:.3f}")
    weights = [args.ind_weight] if args.ind_weight is not None else [0.0, 1.0]
    for wgt in weights:
        kw = {"ind_weight": wgt, "holdover_d": args.holdover_d}
        exp = seat_distribution(races, 0.0, wgt) @ np.arange(len(races) + 1)
        print(f"\nindependents count as D with weight {wgt:g}; expected D seats up: {exp:.2f}")
        print("  rho    P(control)")
        for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            print(f"  {r:4.2f}   {control_prob(races, r, **kw):.4f}")
        for name, target in (("bid", bid), ("mid", mid), ("ask", ask)):
            print(f"  implied rho at {name} {target:.3f}: {_fmt(implied_rhos(races, target, **kw))}")

        swing = {f"SENATE{s.strip().upper()}-26" for s in args.swing.split(",")}
        print(f"  two tiers, swing = {args.swing}: rho among swing pairs -> rho for "
              f"all other pairs that matches control at bid / mid / ask")
        for rs in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            base = [implied_base_rho(races, swing, rs, t, **kw) for t in (bid, mid, ask)]
            print(f"    {rs:4.2f} -> " + " / ".join("  none" if np.isnan(b) else f"{b:6.3f}" for b in base))


if __name__ == "__main__":
    main()
