"""Tests for backtest/engine.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.backtest.engine import (
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backtest_df(
    n_symbols: int = 10,
    n_dates: int = 5,
    base_fwd_return: float = 0.001,
) -> pd.DataFrame:
    """Synthetic df with evenly spaced cs_rank in [-1, 1] per date."""
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B", tz="UTC")
    ranks = np.linspace(-1.0, 1.0, n_symbols)
    rows = []
    for dt in dates:
        for i, sym in enumerate([f"SYM{j}" for j in range(n_symbols)]):
            rows.append(
                {
                    "ts_utc": dt,
                    "symbol": sym,
                    "cs_rank": ranks[i],
                    # Symbols with higher rank get slightly higher returns
                    "fwd_return_1d": base_fwd_return * (i - n_symbols / 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Position assignment
# ---------------------------------------------------------------------------


def test_long_positions_only_in_top_quintile():
    df = _make_backtest_df(n_symbols=10)
    # Add _position column for inspection by computing weights manually
    # Instead, check that portfolio return benefits from high-ranked symbols

    # Verify: for our 10-symbol setup, cs_rank values are:
    # [-1, -0.78, -0.56, -0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1.0]
    # Long threshold = 0.6 → longs: 0.78 and 1.0 (SYM8, SYM9)
    # Set all fwd_returns to 0 except for longs (positive) and shorts (negative)
    df = df.copy()
    df["fwd_return_1d"] = 0.0
    df.loc[df["cs_rank"] > LONG_THRESHOLD, "fwd_return_1d"] = 0.01
    df.loc[df["cs_rank"] < SHORT_THRESHOLD, "fwd_return_1d"] = 0.01  # same for shorts

    returns = run_backtest(df)

    # Longs earn +0.01 with weight +0.5 each = +0.005 + 0.005 = +0.01 total
    # Shorts earn +0.01 with weight -0.5 each = -0.005 - 0.005 = -0.01 total
    # Net = 0 → portfolio return should be 0
    np.testing.assert_allclose(returns.values, 0.0, atol=1e-10)


def test_short_positions_only_in_bottom_quintile():
    df = _make_backtest_df(n_symbols=10)
    df = df.copy()
    # Longs return +1, shorts return 0
    df["fwd_return_1d"] = 0.0
    df.loc[df["cs_rank"] > LONG_THRESHOLD, "fwd_return_1d"] = 1.0
    # Shorts have 0 return → portfolio earns only from longs
    returns = run_backtest(df)
    assert (returns > 0).all()


# ---------------------------------------------------------------------------
# Dollar neutrality
# ---------------------------------------------------------------------------


def test_weights_sum_to_zero_dollar_neutral():
    """If all symbols return the same amount, portfolio return is zero."""
    df = _make_backtest_df(n_symbols=10)
    df["fwd_return_1d"] = 0.05  # identical for all symbols

    returns = run_backtest(df)
    # Long side: +1 × Σ weights_long = +1 × 1.0 = +0.05 gross long
    # Short side: -1 × Σ weights_short = -1.0  → contribution -0.05
    # Net = 0
    np.testing.assert_allclose(returns.values, 0.0, atol=1e-10)


def test_opposite_returns_amplified():
    """Longs +r and shorts -r → portfolio earns 2r."""
    df = _make_backtest_df(n_symbols=10)
    df = df.copy()
    df["fwd_return_1d"] = 0.0
    df.loc[df["cs_rank"] > LONG_THRESHOLD, "fwd_return_1d"] = 0.02   # longs up
    df.loc[df["cs_rank"] < SHORT_THRESHOLD, "fwd_return_1d"] = -0.02  # shorts down

    returns = run_backtest(df)
    # 2 longs  (weight +0.5 each): 2 × 0.5 × (+0.02) = +0.02
    # 2 shorts (weight -0.5 each): 2 × (-0.5) × (-0.02) = +0.02
    # Total per day = 0.04
    np.testing.assert_allclose(returns.values, 0.04, atol=1e-10)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_is_series():
    df = _make_backtest_df()
    result = run_backtest(df)
    assert isinstance(result, pd.Series)


def test_output_indexed_by_ts_utc():
    df = _make_backtest_df(n_dates=3)
    result = run_backtest(df)
    assert result.index.name == "ts_utc"
    assert len(result) == 3


def test_output_dtype_float():
    result = run_backtest(_make_backtest_df())
    assert result.dtype in (np.float64, float)


def test_output_named_portfolio_return():
    result = run_backtest(_make_backtest_df())
    assert result.name == "portfolio_return"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_valid_signals_returns_zero():
    """When all cs_rank values are NaN, portfolio return should be 0."""
    df = _make_backtest_df(n_symbols=5)
    df["cs_rank"] = np.nan
    result = run_backtest(df)
    np.testing.assert_allclose(result.values, 0.0, atol=1e-10)


def test_nan_fwd_returns_treated_as_zero():
    """NaN forward returns (last day) must not distort the portfolio return."""
    df = _make_backtest_df(n_symbols=10, n_dates=2)
    df.loc[df["ts_utc"] == df["ts_utc"].max(), "fwd_return_1d"] = np.nan
    result = run_backtest(df)
    assert pd.isna(result).sum() == 0, "portfolio_return should never be NaN"
