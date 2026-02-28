"""Tests for the canonical normalizer and writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_data_platform.canonical.normalizer import normalize
from market_data_platform.canonical.writer import write_canonical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df(
    symbol: str = "AAPL",
    n: int = 5,
    start: str = "2024-01-01",
    add_dupes: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_utc": dates,
            "symbol": symbol,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000.0 + i * 500 for i in range(n)],
            "ingested_at_utc": pd.Timestamp.now("UTC"),
            "source": "test",
        }
    )
    if add_dupes:
        df = pd.concat([df, df.head(2)], ignore_index=True)
    return df


def _write_raw(df: pd.DataFrame, raw_root: Path) -> None:
    for date_val, group in df.groupby(df["ts_utc"].dt.date):
        symbol = group["symbol"].iloc[0]
        out_dir = raw_root / f"symbol={symbol}" / f"date={date_val}"
        out_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(out_dir / f"part-{date_val}.parquet", index=False)


# ---------------------------------------------------------------------------
# normalize() tests
# ---------------------------------------------------------------------------


def test_normalize_returns_canonical_columns(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(_make_raw_df("AAPL"), raw_root)
    df = normalize(raw_root)
    assert set(df.columns) == {"ts_utc", "symbol", "open", "high", "low", "close", "volume"}


def test_normalize_row_count_without_dupes(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(_make_raw_df("AAPL", n=5), raw_root)
    df = normalize(raw_root)
    assert len(df) == 5


def test_normalize_deduplicates(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(_make_raw_df("AAPL", n=5, add_dupes=True), raw_root)
    df = normalize(raw_root)
    assert len(df) == 5  # duplicates removed


def test_normalize_sorts_by_symbol_then_ts(tmp_path):
    raw_root = tmp_path / "raw"
    for sym in ["MSFT", "AAPL"]:
        _write_raw(_make_raw_df(sym, n=3), raw_root)
    df = normalize(raw_root)
    symbols = df["symbol"].tolist()
    assert symbols == sorted(symbols)
    # Within each symbol timestamps are ascending
    for sym, group in df.groupby("symbol"):
        assert group["ts_utc"].is_monotonic_increasing


def test_normalize_filter_by_symbol(tmp_path):
    raw_root = tmp_path / "raw"
    for sym in ["AAPL", "MSFT"]:
        _write_raw(_make_raw_df(sym, n=3), raw_root)
    df = normalize(raw_root, symbol="AAPL")
    assert set(df["symbol"]) == {"AAPL"}


def test_normalize_raises_if_raw_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw root not found"):
        normalize(tmp_path / "nonexistent")


def test_normalize_raises_if_symbol_missing(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw(_make_raw_df("AAPL"), raw_root)
    with pytest.raises(FileNotFoundError, match="No raw data for symbol"):
        normalize(raw_root, symbol="TSLA")


def test_normalize_raises_if_no_parquet_files(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    with pytest.raises(ValueError, match="No parquet files found"):
        normalize(raw_root)


# ---------------------------------------------------------------------------
# write_canonical() tests
# ---------------------------------------------------------------------------


def test_write_canonical_creates_parquet_files(tmp_path):
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    _write_raw(_make_raw_df("AAPL", n=3), raw_root)
    df = normalize(raw_root)
    n_files = write_canonical(df, canonical_root=canonical_root)
    assert n_files == 3  # 3 daily partitions
    written = list(canonical_root.rglob("*.parquet"))
    assert len(written) == 3


def test_write_canonical_partition_structure(tmp_path):
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    _write_raw(_make_raw_df("AAPL", n=1, start="2024-03-15"), raw_root)
    df = normalize(raw_root)
    write_canonical(df, canonical_root=canonical_root)
    expected = canonical_root / "symbol=AAPL" / "date=2024-03-15" / "part-2024-03-15.parquet"
    assert expected.exists()


def test_write_canonical_is_idempotent(tmp_path):
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    _write_raw(_make_raw_df("AAPL", n=3), raw_root)
    df = normalize(raw_root)
    write_canonical(df, canonical_root=canonical_root)
    # Second write should not raise and should produce the same files
    write_canonical(df, canonical_root=canonical_root)
    written = list(canonical_root.rglob("*.parquet"))
    assert len(written) == 3


def test_write_canonical_empty_df_returns_zero(tmp_path):
    canonical_root = tmp_path / "canonical"
    df = pd.DataFrame(
        columns=["ts_utc", "symbol", "open", "high", "low", "close", "volume"]
    )
    result = write_canonical(df, canonical_root=canonical_root)
    assert result == 0


def test_write_canonical_roundtrip(tmp_path):
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    original = _make_raw_df("AAPL", n=5)
    _write_raw(original, raw_root)
    df = normalize(raw_root)
    write_canonical(df, canonical_root=canonical_root)

    # Read back — read each partition file individually to avoid pyarrow schema
    # merge errors that occur when symbol is encoded as dict vs string across files.
    files = sorted(canonical_root.rglob("*.parquet"))
    reloaded = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    reloaded = reloaded.sort_values("ts_utc").reset_index(drop=True)
    df_sorted = df.sort_values("ts_utc").reset_index(drop=True)
    assert len(reloaded) == len(df_sorted)
    pd.testing.assert_series_equal(
        reloaded["close"].reset_index(drop=True),
        df_sorted["close"].reset_index(drop=True),
        check_names=False,
    )
