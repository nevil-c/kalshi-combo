import numpy as np
import pandas as pd
import pytest

from score import combo_outcome, read_results, score


def _df():
    return pd.DataFrame({
        "leg_markets": ["A-D|B-D", "A-D|B-R", "A-R|C-D", "B-D|C-D"],
        "leg_probs": ["0.8 0.6", "0.8 0.4", "0.2 0.5", "0.6 0.5"],
        "rho": 0.38,
        "engine": [0.55, 0.25, 0.08, 0.35],
        "maker_yes_bid": [0.60, 0.20, np.nan, 0.30],
        "maker_yes_ask": [0.75, 0.24, 0.12, 0.45],
        "maker_mid": [0.675, 0.22, np.nan, 0.375],
    })


def test_combo_outcome():
    res = {"A-D": "yes", "B-D": "no"}
    assert combo_outcome("A-D|B-D", res) == 0.0  # any leg no -> no
    assert np.isnan(combo_outcome("A-D|C-D", res))  # C undecided
    assert combo_outcome("A-D", res) == 1.0


def test_score_metrics():
    res = {"A-D": "yes", "A-R": "no", "B-D": "yes", "B-R": "no", "C-D": "no"}
    s = score(_df(), res)
    assert s["resolved"] == 4 and s["two_sided_resolved"] == 3
    y = np.array([1.0, 0.0, 0.0])
    assert s["brier_engine"] == pytest.approx(np.mean((np.array([0.55, 0.25, 0.35]) - y) ** 2))
    assert s["brier_maker_mid"] == pytest.approx(np.mean((np.array([0.675, 0.22, 0.375]) - y) ** 2))
    assert s["brier_independent"] == pytest.approx(np.mean((np.array([0.48, 0.32, 0.30]) - y) ** 2))


def test_score_with_nothing_resolved():
    s = score(_df(), {})
    assert s["resolved"] == 0 and np.isnan(s["brier_engine"])


def test_read_results_rejects_bad_values(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("market_ticker,result\nA-D,YES\nB-D, no\n")
    assert read_results(p) == {"A-D": "yes", "B-D": "no"}
    p.write_text("market_ticker,result\nA-D,maybe\n")
    with pytest.raises(ValueError):
        read_results(p)
