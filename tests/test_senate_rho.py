import itertools

import numpy as np
import pytest

from pricing import combo_prob
from senate_rho import Race, control_prob, implied_rhos, race_from_markets, seat_distribution


def _races():
    return [Race("A", 0.3, 0.7), Race("B", 0.6, 0.4), Race("C", 0.5, 0.1, 0.4),
            Race("D", 0.9, 0.1), Race("E", 0.2, 0.8)]


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.9, 0.99])
def test_distribution_sums_to_one(rho):
    assert seat_distribution(_races(), rho).sum() == pytest.approx(1.0, abs=1e-9)


def test_rho_zero_matches_brute_force():
    races = _races()
    q = [r.d_seat_share(0.0) * (1 - r.p_r) for r in races]
    brute = np.zeros(len(races) + 1)
    for won in itertools.product([0, 1], repeat=len(races)):
        brute[sum(won)] += np.prod([qi if w else 1 - qi for qi, w in zip(q, won)])
    assert seat_distribution(races, 0.0) == pytest.approx(brute)


@pytest.mark.parametrize("rho", [0.2, 0.6, 0.85, 0.97])
def test_two_races_match_combo_engine(rho):
    races = [Race("A", 0.3, 0.7), Race("B", 0.45, 0.55)]
    both = control_prob(races, rho, holdover_d=0, needed=2)
    assert both == pytest.approx(combo_prob([0.7, 0.55], ["yes", "yes"], rho), abs=1e-7)


def test_independent_weight():
    races = _races()
    lo = control_prob(races, 0.5, ind_weight=0.0, holdover_d=0, needed=3)
    hi = control_prob(races, 0.5, ind_weight=1.0, holdover_d=0, needed=3)
    assert hi > lo


@pytest.mark.parametrize("rho", [0.15, 0.5, 0.8])
def test_implied_rho_roundtrip(rho):
    races = _races()
    target = control_prob(races, rho, holdover_d=0, needed=3)
    assert any(abs(r - rho) < 1e-4 for r in implied_rhos(races, target, holdover_d=0, needed=3))


def test_race_from_markets_normalizes_mids():
    ms = [{"ticker": "SENATENE-26-R", "yes_bid_dollars": "0.69", "yes_ask_dollars": "0.70", "status": "active"},
          {"ticker": "SENATENE-26-D", "yes_bid_dollars": "0.00", "yes_ask_dollars": "0.006", "status": "active"},
          {"ticker": "SENATENE-26-DOSB", "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.31", "status": "active"}]
    r = race_from_markets("SENATENE-26", ms)
    assert r.p_r + r.p_d + r.p_ind == pytest.approx(1.0)
    assert r.p_ind == pytest.approx(0.305 / (0.695 + 0.003 + 0.305))
