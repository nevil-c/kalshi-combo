import math

import numpy as np
import pytest

from pricing import combo_prob
from rfq_scan import (Leg, combo_key, engine_price, enumerate_combos, implied_rho,
                      result_row, run_wave)


def _races():
    return {
        "SENATEAA-26": [Leg("SENATEAA-26", "SENATEAA-26-R", "R a", 0.3),
                        Leg("SENATEAA-26", "SENATEAA-26-D", "D a", 0.7)],
        "SENATEBB-26": [Leg("SENATEBB-26", "SENATEBB-26-R", "R b", 0.6),
                        Leg("SENATEBB-26", "SENATEBB-26-D", "D b", 0.4)],
        "SENATECC-26": [Leg("SENATECC-26", "SENATECC-26-R", "R c", 0.8),
                        Leg("SENATECC-26", "SENATECC-26-D", "D c", 0.1),
                        Leg("SENATECC-26", "SENATECC-26-IND", "Ind c", 0.1)],
        "SENATEDD-26": [Leg("SENATEDD-26", "SENATEDD-26-R", "R d", 0.5),
                        Leg("SENATEDD-26", "SENATEDD-26-D", "D d", 0.5)],
    }


def test_enumerates_one_market_per_race():
    races = _races()
    combos = enumerate_combos(races)
    sizes = [len(v) for v in races.values()]
    expected = sum(math.prod(t) for t in __import__("itertools").combinations(sizes, 3))
    assert len(combos) == expected == 2 * 2 * 3 + 2 * 2 * 2 + 2 * 3 * 2 + 2 * 3 * 2
    assert all(len({l.event_ticker for l in c}) == 3 for c in combos)
    assert len({combo_key(c) for c in combos}) == len(combos)


def test_rho_zero_is_product_of_leg_probs():
    legs = (_races()["SENATEAA-26"][0], _races()["SENATEBB-26"][1], _races()["SENATECC-26"][2])
    assert engine_price(legs, 0.0) == pytest.approx(0.3 * 0.4 * 0.1)


def test_republican_and_democratic_legs_are_negatively_correlated():
    r_a, d_b = _races()["SENATEAA-26"][0], _races()["SENATEBB-26"][1]
    d_a = _races()["SENATEAA-26"][1]
    mixed = [engine_price((r_a, d_b, _races()["SENATEDD-26"][1]), r) for r in (0.0, 0.5)]
    same = [engine_price((d_a, d_b, _races()["SENATEDD-26"][1]), r) for r in (0.0, 0.5)]
    assert mixed[1] < mixed[0]  # R + D legs: correlation lowers P(all win)
    assert same[1] > same[0]    # all anti-R legs: correlation raises it


def test_engine_matches_pricing_module_sides():
    r_a, d_b, ind_c = _races()["SENATEAA-26"][0], _races()["SENATEBB-26"][1], _races()["SENATECC-26"][2]
    assert engine_price((r_a, d_b, ind_c), 0.4) == pytest.approx(
        combo_prob([0.7, 0.4, 0.1], ["no", "yes", "yes"], 0.4))


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.8])
def test_implied_rho_roundtrip(rho):
    legs = (_races()["SENATEAA-26"][1], _races()["SENATEBB-26"][1], _races()["SENATEDD-26"][1])
    assert implied_rho(legs, engine_price(legs, rho)) == pytest.approx(rho, abs=1e-5)
    assert np.isnan(implied_rho(legs, np.nan))


class FakeClient:
    def __init__(self, fail_on_rfq=None):
        self.calls, self.open, self.fail_on_rfq, self.n = [], set(), fail_on_rfq, 0

    def request(self, method, path, params=None, body=None):
        self.calls.append((method, path, body))
        if method == "POST" and path.startswith("/multivariate_event_collections/"):
            return {"market_ticker": "COMBO-" + "|".join(m["market_ticker"] for m in body["selected_markets"])}
        if method == "POST" and path == "/communications/rfqs":
            self.n += 1
            if self.fail_on_rfq and self.n == self.fail_on_rfq:
                raise RuntimeError("boom")
            assert body["rest_remainder"] is False
            rfq_id = f"rfq{self.n}"
            self.open.add(rfq_id)
            return {"id": rfq_id}
        if method == "DELETE":
            self.open.discard(path.rsplit("/", 1)[-1])
            return {}
        if method == "GET" and path == "/communications/quotes":
            return {"quotes": [{"yes_bid_dollars": "0.10", "no_bid_dollars": "0.85"},
                               {"yes_bid_dollars": "0.11", "no_bid_dollars": "0"}]}
        raise AssertionError(f"unexpected call {method} {path}")

    def get(self, path, **params):
        return self.request("GET", path, params)


def test_wave_collects_quotes_deletes_rfqs_and_never_accepts(tmp_path):
    client = FakeClient()
    wave = enumerate_combos(_races())[:5]
    cache = {}
    res = run_wave(client, wave, cache, tmp_path / "combos.csv", 100, 0, log=lambda *_: None)
    assert len(res) == 5 and not client.open
    assert all(r["best_yes_bid"] == 0.11 and r["best_no_bid"] == 0.85 for r in res)
    assert not any("accept" in p or "confirm" in p for _, p, _ in client.calls)
    # combos are cached: a second wave creates nothing new
    client2 = FakeClient()
    run_wave(client2, wave, cache, tmp_path / "combos.csv", 100, 0, log=lambda *_: None)
    assert not any(p.startswith("/multivariate") for _, p, _ in client2.calls)


def test_open_rfqs_deleted_when_wave_fails(tmp_path):
    client = FakeClient(fail_on_rfq=3)
    with pytest.raises(RuntimeError):
        run_wave(client, enumerate_combos(_races())[:5], {}, tmp_path / "c.csv", 100, 0,
                 log=lambda *_: None)
    assert client.n == 3 and not client.open


def test_result_row_maker_mid_and_implied_rho():
    legs = enumerate_combos(_races())[0]
    row = result_row({"legs": legs, "combo_ticker": "C", "rfq_id": "x", "n_quotes": 2,
                      "best_yes_bid": 0.10, "best_no_bid": 0.85, "quoted_at": "t"}, rho=0.3)
    assert row["maker_yes_ask"] == pytest.approx(0.15)
    assert row["maker_mid"] == pytest.approx(0.125)
    assert row["engine_minus_mid"] == pytest.approx(engine_price(legs, 0.3) - 0.125)
