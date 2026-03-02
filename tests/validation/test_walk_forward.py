"""Tests for validation/walk_forward.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.validation.walk_forward import WalkForwardSplitter

# ---------------------------------------------------------------------------
# Compact splitter config for fast tests (30-day train, 10-day test, 10-step)
# ---------------------------------------------------------------------------

_SP = dict(train_period=30, test_period=10, step_size=10, min_train=25)


def _make_df(n_dates: int = 60, n_symbols: int = 3) -> pd.DataFrame:
    """Minimal panel DataFrame with ts_utc column and n_symbols symbols."""
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B", tz="UTC")
    rows = [
        {"ts_utc": dt, "symbol": f"SYM{i}", "value": 1.0}
        for dt in dates
        for i in range(n_symbols)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------


def test_split_returns_correct_n_windows():
    """60 dates → 3 non-overlapping test windows (30 train, 10 test, step 10)."""
    # Window 0: train dates[0:30], test dates[30:40]
    # Window 1: train dates[10:40], test dates[40:50]
    # Window 2: train dates[20:50], test dates[50:60]
    # Window 3 would need test dates[60:70] — none left → stops
    df = _make_df(n_dates=60)
    splits = WalkForwardSplitter(**_SP).split(df)
    assert len(splits) == 3


def test_n_splits_property_matches_len_splits():
    """n_splits property equals the length of the returned splits list."""
    df = _make_df(n_dates=60)
    splitter = WalkForwardSplitter(**_SP)
    splits = splitter.split(df)
    assert splitter.n_splits == len(splits)


# ---------------------------------------------------------------------------
# Leakage and overlap guarantees
# ---------------------------------------------------------------------------


def test_no_data_leakage():
    """Every test window starts strictly after its training window ends."""
    df = _make_df(n_dates=60)
    for train_df, test_df in WalkForwardSplitter(**_SP).split(df):
        assert test_df["ts_utc"].min() > train_df["ts_utc"].max()


def test_train_test_no_overlap():
    """No single date appears in both train and test for any window."""
    df = _make_df(n_dates=60)
    for train_df, test_df in WalkForwardSplitter(**_SP).split(df):
        train_dates = set(train_df["ts_utc"].unique())
        test_dates = set(test_df["ts_utc"].unique())
        assert train_dates.isdisjoint(test_dates)


# ---------------------------------------------------------------------------
# min_train enforcement
# ---------------------------------------------------------------------------


def test_min_train_respected():
    """Windows with fewer training days than min_train are skipped entirely."""
    df = _make_df(n_dates=60)
    # min_train=31 > train_period=30 → every window is below min → zero splits
    splitter = WalkForwardSplitter(train_period=30, test_period=10, step_size=10, min_train=31)
    splits = splitter.split(df)
    assert len(splits) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_symbol_works():
    """Splitter handles a single-symbol DataFrame correctly."""
    df = _make_df(n_dates=60, n_symbols=1)
    splits = WalkForwardSplitter(**_SP).split(df)
    assert len(splits) == 3


def test_empty_df_returns_no_splits():
    """Empty DataFrame produces an empty splits list without raising."""
    df = pd.DataFrame({"ts_utc": pd.Series([], dtype="datetime64[ns, UTC]"), "symbol": []})
    splits = WalkForwardSplitter(**_SP).split(df)
    assert splits == []


def test_train_period_exactly_fills_window():
    """With n_dates == train_period + test_period we get exactly 1 split."""
    df = _make_df(n_dates=40)
    splits = WalkForwardSplitter(**_SP).split(df)
    assert len(splits) == 1
    train_df, test_df = splits[0]
    assert train_df["ts_utc"].nunique() == 30
    assert test_df["ts_utc"].nunique() == 10


def test_insufficient_dates_returns_no_splits():
    """Fewer dates than train_period → no splits possible."""
    df = _make_df(n_dates=20)
    splits = WalkForwardSplitter(**_SP).split(df)
    assert len(splits) == 0


# ---------------------------------------------------------------------------
# Step-size behaviour
# ---------------------------------------------------------------------------


def test_step_size_advances_correctly():
    """Consecutive test windows start exactly step_size trading days apart."""
    df = _make_df(n_dates=60)
    splitter = WalkForwardSplitter(**_SP)
    splits = splitter.split(df)
    assert len(splits) >= 2

    all_dates = sorted(df["ts_utc"].unique())
    date_idx = {d: i for i, d in enumerate(all_dates)}
    test_starts = [sorted(test_df["ts_utc"].unique())[0] for _, test_df in splits]

    for i in range(1, len(test_starts)):
        gap = date_idx[test_starts[i]] - date_idx[test_starts[i - 1]]
        assert gap == splitter.step_size


# ---------------------------------------------------------------------------
# DatetimeIndex support
# ---------------------------------------------------------------------------


def test_datetime_index_support():
    """Splitter accepts a df with DatetimeIndex named ts_utc (no ts_utc column)."""
    dates = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    df = pd.DataFrame({"value": np.ones(60)}, index=dates)
    df.index.name = "ts_utc"
    splits = WalkForwardSplitter(**_SP).split(df)
    assert len(splits) == 3


def test_invalid_df_raises():
    """DataFrame without ts_utc column or DatetimeIndex raises ValueError."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-02", periods=10), "value": range(10)})
    with pytest.raises(ValueError, match="ts_utc"):
        WalkForwardSplitter(**_SP).split(df)
