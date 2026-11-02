import pandas as pd
import pytest

from calibrate import CELLS, COMBO_EVENT
from compare import compare
from parse import load_snapshot
from pathlib import Path
from pricing import combo_prob


def _quotes(p_h, p_s, rho, half_spread=0.005):
    rows = [("CONTROLH-2026-D", p_h), ("CONTROLH-2026-R", 1 - p_h),
            ("CONTROLS-2026-D", p_s), ("CONTROLS-2026-R", 1 - p_s)]
    rows += [(f"{COMBO_EVENT}-{k}", combo_prob([p_h, p_s], s, rho)) for k, s in CELLS.items()]
    df = pd.DataFrame(rows, columns=["ticker", "mid"])
    return df.assign(bid=df.mid - half_spread, ask=df.mid + half_spread)


def test_engine_matches_quotes_at_true_rho():
    table, info = compare(_quotes(0.9, 0.6, 0.5), rho=0.5)
    assert not info["rho_fitted"]
    assert table["engine_minus_mid"].abs().max() < 1e-12
    assert (table["vs_quote"] == "inside").all()
    assert table["engine"].sum() == pytest.approx(1.0, abs=1e-9)


def test_fitted_rho_is_in_sample():
    _, info = compare(_quotes(0.9, 0.6, 0.5))
    assert info["rho_fitted"]
    assert info["rho"] == pytest.approx(0.5, abs=1e-4)


def test_flags_prices_outside_the_quote():
    table, _ = compare(_quotes(0.9, 0.6, 0.5), rho=0.0)
    # independence underprices the DD sweep and overprices the split DR cell
    assert table.loc["DD", "vs_quote"] == "below bid"
    assert table.loc["DR", "vs_quote"] == "above ask"


def test_load_snapshot_single_file():
    df = load_snapshot(Path(__file__).parent / "fixtures" / "snapshot_20261027T000000Z.json")
    assert len(df) == 3 and df["time"].nunique() == 1
