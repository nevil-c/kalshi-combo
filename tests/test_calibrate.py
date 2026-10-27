import pandas as pd
import pytest

from calibrate import CELLS, COMBO_EVENT, calibrate_snapshot, implied_rho, leg_prob
from pricing import combo_prob


def _cells(p_h, p_s, rho):
    return {k: combo_prob([p_h, p_s], s, rho) for k, s in CELLS.items()}


@pytest.mark.parametrize("rho", [0.3, 0.7])
@pytest.mark.parametrize("p_h,p_s", [(0.907, 0.64), (0.5, 0.5), (0.2, 0.75)])
def test_recovers_known_rho(p_h, p_s, rho):
    res = implied_rho(p_h, p_s, _cells(p_h, p_s, rho))
    assert res.rho == pytest.approx(rho, abs=1e-4)
    assert res.table["resid"].abs().max() < 1e-6
    assert res.overround == pytest.approx(0.0, abs=1e-9)
    assert res.table["cell_rho"].dropna().sub(rho).abs().max() < 1e-4


def test_leg_prob_uses_both_sides():
    quotes = {"D": 0.90, "R": 0.12}
    assert leg_prob(quotes, "D", "R") == pytest.approx(0.89)


def test_calibrate_snapshot_from_quotes():
    p_h, p_s, rho = 0.9, 0.6, 0.5
    rows = [("CONTROLH-2026-D", p_h), ("CONTROLH-2026-R", 1 - p_h),
            ("CONTROLS-2026-D", p_s), ("CONTROLS-2026-R", 1 - p_s)]
    rows += [(f"{COMBO_EVENT}-{k}", v) for k, v in _cells(p_h, p_s, rho).items()]
    df = pd.DataFrame(rows, columns=["ticker", "mid"]).assign(bid=lambda d: d.mid - 0.005,
                                                             ask=lambda d: d.mid + 0.005)
    assert calibrate_snapshot(df).rho == pytest.approx(rho, abs=1e-4)
    assert calibrate_snapshot(df, weighted=True).rho == pytest.approx(rho, abs=1e-4)
