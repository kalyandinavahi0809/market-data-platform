"""Tests for research/forward_returns.py.

Key invariants verified
-----------------------
1. fwd_return_1d[t] == log_return_1d[t+1]   (exact identity)
2. fwd_return_5d[t] == log_return_5d[t+5]   (exact identity)
3. Existing feature columns are not modified
4. Last row(s) have NaN forward returns
5. No forward-looking data leaks into feature columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.research.features import compute_features
from market_data_platform.research.forward_returns import add_forward_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 20, symbol: str = "AAPL") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = [100.0 + 0.5 * i for i in range(n)]
    return pd.DataFrame(
        {
            "ts_utc": dates,
            "symbol": symbol,
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * n,
        }
    )


def _get_sym(df: pd.DataFrame, symbol: str = "AAPL") -> pd.DataFrame:
    return df[df["symbol"] == symbol].sort_values("ts_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# fwd_return_1d == log_return_1d at t+1
# ---------------------------------------------------------------------------


def test_fwd_return_1d_equals_log_return_1d_at_t_plus_1():
    df = _make_df(n=15)
    features = compute_features(df)
    result = add_forward_returns(features)

    sym = _get_sym(result)

    # fwd_return_1d[i] must equal log_return_1d[i+1]
    pd.testing.assert_series_equal(
        sym["fwd_return_1d"].iloc[:-1].reset_index(drop=True),
        sym["log_return_1d"].iloc[1:].reset_index(drop=True),
        check_names=False,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# fwd_return_5d == log_return_5d at t+5
# ---------------------------------------------------------------------------


def test_fwd_return_5d_equals_log_return_5d_at_t_plus_5():
    df = _make_df(n=20)
    features = compute_features(df)
    result = add_forward_returns(features)

    sym = _get_sym(result)

    pd.testing.assert_series_equal(
        sym["fwd_return_5d"].iloc[:-5].reset_index(drop=True),
        sym["log_return_5d"].iloc[5:].reset_index(drop=True),
        check_names=False,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# NaN at series end
# ---------------------------------------------------------------------------


def test_fwd_return_1d_nan_at_last_row():
    df = _make_df(n=10)
    result = add_forward_returns(compute_features(df))
    sym = _get_sym(result)
    assert pd.isna(sym["fwd_return_1d"].iloc[-1])


def test_fwd_return_5d_nan_at_last_five_rows():
    df = _make_df(n=10)
    result = add_forward_returns(compute_features(df))
    sym = _get_sym(result)
    assert sym["fwd_return_5d"].iloc[-5:].isna().all()


# ---------------------------------------------------------------------------
# Feature columns unchanged
# ---------------------------------------------------------------------------


def test_feature_columns_not_modified():
    df = _make_df(n=15)
    features = compute_features(df)
    result = add_forward_returns(features)

    for col in ["log_return_1d", "log_return_5d", "log_return_20d", "vol_20d", "volume_zscore"]:
        pd.testing.assert_series_equal(
            features[col].reset_index(drop=True),
            result[col].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# No forward-looking contamination in feature columns
# ---------------------------------------------------------------------------


def test_no_forward_data_leaks_into_feature_columns():
    """Feature at t must equal feature at t computed on a truncated dataset.

    If add_forward_returns modified feature columns (lookahead), this test
    would fail because the truncated dataset wouldn't have the future data.
    """
    df_full = _make_df(n=20)
    df_trunc = _make_df(n=15)

    result_full = add_forward_returns(compute_features(df_full))
    result_trunc = add_forward_returns(compute_features(df_trunc))

    ts = df_trunc["ts_utc"].iloc[-1]

    for col in ["log_return_1d", "log_return_20d"]:
        full_val = result_full.loc[result_full["ts_utc"] == ts, col].iloc[0]
        trunc_val = result_trunc.loc[result_trunc["ts_utc"] == ts, col].iloc[0]

        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        assert abs(full_val - trunc_val) < 1e-10, (
            f"{col} differs: full={full_val}, trunc={trunc_val}"
        )


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------


def test_forward_returns_independent_per_symbol():
    """fwd_return_1d for AAPL must not be contaminated by MSFT data."""
    df_aapl = _make_df(n=10, symbol="AAPL")
    df_msft = _make_df(n=10, symbol="MSFT")
    df_combined = pd.concat([df_aapl, df_msft], ignore_index=True)

    result_combined = add_forward_returns(compute_features(df_combined))
    result_standalone = add_forward_returns(compute_features(df_aapl))

    combined_aapl = _get_sym(result_combined, "AAPL")
    standalone = _get_sym(result_standalone, "AAPL")

    pd.testing.assert_series_equal(
        combined_aapl["fwd_return_1d"],
        standalone["fwd_return_1d"],
        check_names=False,
        rtol=1e-10,
    )
