"""Tests for data quality checks."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_data_platform.quality.checks import (
    CheckResult,
    QualityReport,
    check_freshness,
    check_gaps,
    check_ohlcv_sanity,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 5,
    start: str = "2024-01-02",
    freq: str = "B",
    symbol: str = "AAPL",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "ts_utc": dates,
            "symbol": symbol,
            "open": [100.0] * n,
            "high": [102.0] * n,
            "low": [98.0] * n,
            "close": [101.0] * n,
            "volume": [500_000.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# check_ohlcv_sanity
# ---------------------------------------------------------------------------


def test_sanity_passes_clean_data():
    result = check_ohlcv_sanity(_make_df(5))
    assert result.passed
    assert result.name == "ohlcv_sanity"


def test_sanity_fails_high_below_low():
    df = _make_df(3)
    df.loc[1, "high"] = 95.0  # below low=98
    result = check_ohlcv_sanity(df)
    assert not result.passed
    assert "high < low" in result.message


def test_sanity_fails_close_above_high():
    df = _make_df(3)
    df.loc[0, "close"] = 200.0  # above high=102
    result = check_ohlcv_sanity(df)
    assert not result.passed
    assert "close > high" in result.message


def test_sanity_fails_close_below_low():
    df = _make_df(3)
    df.loc[0, "close"] = 50.0  # below low=98
    result = check_ohlcv_sanity(df)
    assert not result.passed
    assert "close < low" in result.message


def test_sanity_fails_negative_volume():
    df = _make_df(3)
    df.loc[2, "volume"] = -1.0
    result = check_ohlcv_sanity(df)
    assert not result.passed
    assert "negative volume" in result.message


def test_sanity_fails_null_price():
    df = _make_df(3)
    df.loc[0, "close"] = float("nan")
    result = check_ohlcv_sanity(df)
    assert not result.passed
    assert "NULLs" in result.message


# ---------------------------------------------------------------------------
# check_gaps
# ---------------------------------------------------------------------------


def test_gaps_passes_no_gaps():
    df = _make_df(n=5, start="2024-01-02", freq="B")
    result = check_gaps(df, freq="B")
    assert result.passed


def test_gaps_detects_missing_business_day():
    # 2024-01-02 (Tue), skip 2024-01-03 (Wed), 2024-01-04 (Thu)
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"]).tz_localize("UTC")
    df = _make_df(2)
    df["ts_utc"] = dates
    result = check_gaps(df, freq="B")
    assert not result.passed
    assert "2024-01-03" in result.message


def test_gaps_passes_empty_df():
    df = pd.DataFrame(columns=["ts_utc"])
    result = check_gaps(df)
    assert result.passed


def test_gaps_passes_single_row():
    df = _make_df(1)
    result = check_gaps(df, freq="B")
    assert result.passed


def test_gaps_crypto_uses_calendar_days():
    # Saturday between two Sundays — fine for crypto (freq="D") but a gap for "B"
    dates = pd.to_datetime(["2024-01-06", "2024-01-07"]).tz_localize("UTC")
    df = _make_df(2)
    df["ts_utc"] = dates
    result_daily = check_gaps(df, freq="D")
    assert result_daily.passed


# ---------------------------------------------------------------------------
# check_freshness
# ---------------------------------------------------------------------------


def test_freshness_passes_recent_data():
    ref = date(2024, 1, 10)
    df = _make_df(n=1, start="2024-01-09")  # 1 day old
    result = check_freshness(df, max_staleness_days=3, reference_date=ref)
    assert result.passed


def test_freshness_fails_stale_data():
    ref = date(2024, 1, 10)
    df = _make_df(n=1, start="2024-01-01")  # 9 days old
    result = check_freshness(df, max_staleness_days=3, reference_date=ref)
    assert not result.passed
    assert "9 day(s)" in result.message


def test_freshness_fails_empty_df():
    df = pd.DataFrame(columns=["ts_utc"])
    result = check_freshness(df, reference_date=date(2024, 1, 10))
    assert not result.passed


def test_freshness_passes_on_threshold_boundary():
    ref = date(2024, 1, 10)
    df = _make_df(n=1, start="2024-01-07")  # exactly 3 days old
    result = check_freshness(df, max_staleness_days=3, reference_date=ref)
    assert result.passed


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


def test_quality_report_passed_when_all_checks_pass():
    df = _make_df(5)
    ref = pd.Timestamp(df["ts_utc"].max()).date()
    report = run_all_checks(df, symbol="AAPL", reference_date=ref)
    assert report.passed
    assert report.symbol == "AAPL"


def test_quality_report_failed_when_any_check_fails():
    df = _make_df(5)
    df.loc[0, "high"] = 50.0  # below low
    ref = pd.Timestamp(df["ts_utc"].max()).date()
    report = run_all_checks(df, symbol="AAPL", reference_date=ref)
    assert not report.passed
    assert len(report.failed_checks) >= 1


def test_quality_report_summary_contains_symbol():
    df = _make_df(3)
    ref = pd.Timestamp(df["ts_utc"].max()).date()
    report = run_all_checks(df, symbol="MSFT", reference_date=ref)
    assert "MSFT" in report.summary()
