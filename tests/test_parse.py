from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parse import COLUMNS, latest, load_snapshots

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def df():
    return load_snapshots(FIXTURES)


def test_columns_and_types(df):
    assert list(df.columns) == COLUMNS
    assert len(df) == 4
    assert str(df["time"].dt.tz) == "UTC"
    assert df["bid"].dtype == float and df["ask"].dtype == float


def test_dollar_strings_and_mid(df):
    row = df[(df.ticker == "CONTROLH-2026-D")].iloc[0]
    assert row.bid == pytest.approx(0.908)
    assert row.ask == pytest.approx(0.909)
    assert row.mid == pytest.approx(0.9085)
    assert row.time == pd.Timestamp("2026-10-27T00:00:00Z")


def test_missing_side_gives_nan_mid(df):
    senate = df[df.ticker == "CONTROLS-2026-D"].iloc[0]
    combo = df[df.ticker == "KXBALANCEPOWERCOMBO-27FEB-DD"].iloc[0]
    assert np.isnan(senate.ask) and np.isnan(senate.mid)
    assert np.isnan(combo.bid) and np.isnan(combo.mid)


def test_latest_picks_newest(df):
    last = latest(df).set_index("ticker")
    assert last.loc["CONTROLH-2026-D", "mid"] == pytest.approx(0.911)
    assert len(last) == 3


def test_empty_dir(tmp_path):
    out = load_snapshots(tmp_path)
    assert out.empty and list(out.columns) == COLUMNS
