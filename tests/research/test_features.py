"""Tests for research/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.research.features import compute_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 30,
    symbol: str = "AAPL",
    close_start: float = 100.0,
    close_slope: float = 0.1,
) -> pd.DataFrame:
    """Single-symbol canonical OHLCV with linearly increasing close prices."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = [close_start + close_slope * i for i in range(n)]
    return pd.DataFrame(
        {
            "ts_utc": dates,
            "symbol": symbol,
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1_000_000.0 + i * 500 for i in range(n)],
        }
    )


def _make_multi_df(n_symbols: int = 3, n: int = 30) -> pd.DataFrame:
    frames = [
        _make_df(n=n, symbol=f"SYM{i}", close_start=100.0 + i * 10, close_slope=0.1 * (i + 1))
        for i in range(n_symbols)
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Correctness of individual features
# ---------------------------------------------------------------------------


def test_log_return_1d_correct():
    df = _make_df(n=5)
    result = compute_features(df)
    closes = df.sort_values("ts_utc")["close"].values
    expected = np.log(closes[1] / closes[0])
    assert abs(result["log_return_1d"].iloc[1] - expected) < 1e-10


def test_log_return_5d_correct():
    df = _make_df(n=10)
    result = compute_features(df)
    closes = df.sort_values("ts_utc")["close"].values
    expected = np.log(closes[5] / closes[0])
    assert abs(result["log_return_5d"].iloc[5] - expected) < 1e-10


def test_log_return_20d_correct():
    df = _make_df(n=25)
    result = compute_features(df)
    closes = df.sort_values("ts_utc")["close"].values
    expected = np.log(closes[20] / closes[0])
    assert abs(result["log_return_20d"].iloc[20] - expected) < 1e-10


def test_vol_20d_correct():
    df = _make_df(n=30)
    result = compute_features(df)
    # Manually compute: rolling(20) std of log_return_1d at index 20
    log_returns = result["log_return_1d"].values
    window = log_returns[1:21]  # 20 valid values
    expected_std = np.std(window, ddof=1) * np.sqrt(252)
    assert abs(result["vol_20d"].iloc[20] - expected_std) < 1e-10


def test_volume_zscore_correct():
    df = _make_df(n=30)
    result = compute_features(df)
    # At index 25: z-score uses volumes[6..25] (20 observations)
    vols = df.sort_values("ts_utc")["volume"].values
    window = vols[6:26]
    expected_z = (vols[25] - window.mean()) / np.std(window, ddof=1)
    assert abs(result["volume_zscore"].iloc[25] - expected_z) < 1e-10


# ---------------------------------------------------------------------------
# NaN boundaries
# ---------------------------------------------------------------------------


def test_log_return_1d_nan_at_first_row():
    result = compute_features(_make_df(n=5))
    assert pd.isna(result["log_return_1d"].iloc[0])


def test_log_return_5d_nan_for_first_five_rows():
    result = compute_features(_make_df(n=10))
    assert result["log_return_5d"].iloc[:5].isna().all()
    assert not pd.isna(result["log_return_5d"].iloc[5])


def test_log_return_20d_nan_for_first_twenty_rows():
    result = compute_features(_make_df(n=25))
    assert result["log_return_20d"].iloc[:20].isna().all()
    assert not pd.isna(result["log_return_20d"].iloc[20])


def test_vol_20d_nan_for_first_twenty_rows():
    # rolling(20, min_periods=20) on log_return_1d (which is NaN at index 0):
    # first valid vol_20d is at index 20 (window covers indices 1..20, all valid)
    result = compute_features(_make_df(n=30))
    assert result["vol_20d"].iloc[:20].isna().all()
    assert not pd.isna(result["vol_20d"].iloc[20])


# ---------------------------------------------------------------------------
# No-lookahead guarantee
# ---------------------------------------------------------------------------


def test_no_lookahead_features_unchanged_when_future_rows_added():
    """Feature at t must not change when rows after t are appended."""
    df_full = _make_df(n=30)
    df_trunc = _make_df(n=25)

    result_full = compute_features(df_full)
    result_trunc = compute_features(df_trunc)

    # Check the last timestamp of the truncated set (day 24)
    ts = df_trunc["ts_utc"].iloc[-1]
    for col in ["log_return_1d", "log_return_5d", "log_return_20d", "vol_20d", "volume_zscore"]:
        full_val = result_full.loc[result_full["ts_utc"] == ts, col].iloc[0]
        trunc_val = result_trunc.loc[result_trunc["ts_utc"] == ts, col].iloc[0]

        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        assert abs(full_val - trunc_val) < 1e-10, (
            f"{col}: full={full_val}, trunc={trunc_val}"
        )


# ---------------------------------------------------------------------------
# Multi-symbol isolation
# ---------------------------------------------------------------------------


def test_multi_symbol_features_match_standalone():
    """Features for AAPL in a multi-symbol df must match a standalone computation."""
    df_aapl = _make_df(n=25, symbol="AAPL")
    df_msft = _make_df(n=25, symbol="MSFT", close_start=200.0)
    df_combined = pd.concat([df_aapl, df_msft], ignore_index=True)

    result_combined = compute_features(df_combined)
    result_standalone = compute_features(df_aapl)

    combined_aapl = (
        result_combined[result_combined["symbol"] == "AAPL"]
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )
    standalone = result_standalone.sort_values("ts_utc").reset_index(drop=True)

    for col in ["log_return_1d", "log_return_20d", "vol_20d"]:
        pd.testing.assert_series_equal(
            combined_aapl[col],
            standalone[col],
            check_names=False,
            rtol=1e-10,
        )


def test_output_sorted_by_symbol_then_ts():
    result = compute_features(_make_multi_df(n_symbols=3, n=5))
    pairs = list(zip(result["symbol"], result["ts_utc"]))
    assert pairs == sorted(pairs)
