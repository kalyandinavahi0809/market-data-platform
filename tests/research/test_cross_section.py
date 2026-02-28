"""Tests for research/cross_section.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.research.cross_section import compute_cross_section
from market_data_platform.research.features import compute_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multi_df(
    n_symbols: int = 5,
    n_dates: int = 25,
    distinct_slopes: bool = True,
) -> pd.DataFrame:
    """Multi-symbol df with distinct close price trajectories."""
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D", tz="UTC")
    frames = []
    for i in range(n_symbols):
        # Distinct slopes ensure distinct log_return_20d values (no ties)
        slope = 0.05 * (i + 1) if distinct_slopes else 0.1
        closes = [100.0 * (1 + slope * j / 100) for j in range(n_dates)]
        frames.append(
            pd.DataFrame(
                {
                    "ts_utc": dates,
                    "symbol": f"SYM{i}",
                    "open": closes,
                    "high": [c * 1.01 for c in closes],
                    "low": [c * 0.99 for c in closes],
                    "close": closes,
                    "volume": [1_000_000.0] * n_dates,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _features(n_symbols: int = 5, n_dates: int = 25) -> pd.DataFrame:
    return compute_features(_make_multi_df(n_symbols=n_symbols, n_dates=n_dates))


# ---------------------------------------------------------------------------
# Range constraint: cs_rank ∈ [−1, 1]
# ---------------------------------------------------------------------------


def test_ranks_bounded_in_minus1_to_1():
    df = _features(n_symbols=5, n_dates=25)
    result = compute_cross_section(df)
    valid = result["cs_rank"].dropna()
    assert (valid >= -1.0 - 1e-10).all(), "cs_rank below -1"
    assert (valid <= 1.0 + 1e-10).all(), "cs_rank above +1"


def test_cross_sectional_extremes_are_minus1_and_1():
    """The bottom symbol must receive rank -1 and the top must receive +1."""
    df = _features(n_symbols=5, n_dates=25)
    result = compute_cross_section(df)

    # Find dates where all 5 symbols have valid ranks
    date_counts = result.dropna(subset=["cs_rank"]).groupby("ts_utc")["cs_rank"].count()
    full_dates = date_counts[date_counts == 5].index

    if full_dates.empty:
        pytest.skip("No dates with all 5 symbols having valid signals")

    for dt in full_dates[:3]:
        day_ranks = result.loc[result["ts_utc"] == dt, "cs_rank"]
        assert abs(day_ranks.min() - (-1.0)) < 1e-10
        assert abs(day_ranks.max() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Cross-sectional mean ≈ 0
# ---------------------------------------------------------------------------


def test_ranks_mean_zero_per_date():
    """Rank normalization guarantees zero mean cross-sectionally."""
    df = _features(n_symbols=5, n_dates=25)
    result = compute_cross_section(df)

    # Only consider dates where all symbols have valid ranks
    valid_dates = (
        result.dropna(subset=["cs_rank"])
        .groupby("ts_utc")["cs_rank"]
        .apply(lambda s: len(s) == 5)
    )
    full_dates = valid_dates[valid_dates].index

    for dt in full_dates:
        day_ranks = result.loc[result["ts_utc"] == dt, "cs_rank"]
        np.testing.assert_allclose(day_ranks.mean(), 0.0, atol=1e-10)


def test_zscore_mean_approximately_zero_per_date():
    df = _features(n_symbols=5, n_dates=25)
    result = compute_cross_section(df)

    valid_dates = (
        result.dropna(subset=["cs_zscore"])
        .groupby("ts_utc")["cs_zscore"]
        .apply(lambda s: len(s) >= 2)
    )
    full_dates = valid_dates[valid_dates].index

    for dt in full_dates[:5]:
        day_z = result.loc[result["ts_utc"] == dt, "cs_zscore"].dropna()
        np.testing.assert_allclose(day_z.mean(), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_symbol_gets_neutral_rank():
    """A date where only one symbol has a valid signal → rank = 0."""
    # Build a df where AAPL has enough history for log_return_20d but MSFT doesn't
    dates = pd.date_range("2024-01-01", periods=25, freq="D", tz="UTC")
    aapl_df = pd.DataFrame(
        {
            "ts_utc": dates,
            "symbol": "AAPL",
            "open": [100.0] * 25,
            "high": [101.0] * 25,
            "low": [99.0] * 25,
            "close": [100.0 + 0.5 * i for i in range(25)],
            "volume": [1_000_000.0] * 25,
        }
    )
    # MSFT only 5 rows — too short for log_return_20d (needs 21 rows)
    msft_df = pd.DataFrame(
        {
            "ts_utc": dates[:5],
            "symbol": "MSFT",
            "open": [200.0] * 5,
            "high": [201.0] * 5,
            "low": [199.0] * 5,
            "close": [200.0] * 5,
            "volume": [1_000_000.0] * 5,
        }
    )
    df = pd.concat([aapl_df, msft_df], ignore_index=True)
    features = compute_features(df)
    result = compute_cross_section(features)

    # Find a date where only AAPL has valid log_return_20d (from day 21 onward)
    aapl_result = result[(result["symbol"] == "AAPL") & result["cs_rank"].notna()]
    # On dates where MSFT has NaN log_return_20d, AAPL should be the only valid symbol
    dt = aapl_result["ts_utc"].iloc[0]
    msft_row = result[(result["symbol"] == "MSFT") & (result["ts_utc"] == dt)]

    if msft_row.empty or pd.isna(msft_row["log_return_20d"].iloc[0]):
        # AAPL is the only valid symbol → rank should be 0
        aapl_rank = result[(result["symbol"] == "AAPL") & (result["ts_utc"] == dt)]["cs_rank"].iloc[0]
        assert abs(aapl_rank) < 1e-10


def test_nan_signal_gives_nan_rank():
    """Symbols with NaN signal at a timestamp must have NaN cs_rank."""
    df = _features(n_symbols=3, n_dates=25)
    result = compute_cross_section(df)

    nan_signal_mask = df["log_return_20d"].isna()
    assert result.loc[nan_signal_mask, "cs_rank"].isna().all()


def test_handles_two_symbols():
    """With exactly two symbols, ranks should be -1 and +1."""
    df = _features(n_symbols=2, n_dates=25)
    result = compute_cross_section(df)

    valid_dates = (
        result.dropna(subset=["cs_rank"])
        .groupby("ts_utc")["cs_rank"]
        .apply(lambda s: len(s) == 2)
    )
    full_dates = valid_dates[valid_dates].index

    for dt in full_dates[:3]:
        day_ranks = sorted(result.loc[result["ts_utc"] == dt, "cs_rank"].tolist())
        np.testing.assert_allclose(day_ranks, [-1.0, 1.0], atol=1e-10)
