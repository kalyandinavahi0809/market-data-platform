"""Tests for DuckDBClient."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_data_platform.storage.duckdb_client import DuckDBClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet(directory: Path, symbol: str = "AAPL", n: int = 3) -> None:
    """Write a minimal parquet file into *directory* for view registration tests."""
    table = pa.table(
        {
            "ts_utc": pd.date_range("2024-01-01", periods=n, freq="D").tolist(),
            "symbol": [symbol] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000.0 + i * 1000 for i in range(n)],
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "part-0.parquet")


# ---------------------------------------------------------------------------
# Init / lifecycle
# ---------------------------------------------------------------------------


def test_creates_db_file(tmp_path):
    db_path = tmp_path / "test.duckdb"
    client = DuckDBClient(db_path=db_path)
    assert db_path.exists(), "DuckDB file should be created on init"
    assert client.db_path == db_path
    client.close()


def test_in_memory_does_not_create_file(tmp_path):
    client = DuckDBClient(db_path=Path(":memory:"))
    assert not (tmp_path / "market_data.duckdb").exists()
    client.close()


def test_context_manager_closes_connection(tmp_path):
    db_path = tmp_path / "ctx.duckdb"
    with DuckDBClient(db_path=db_path) as client:
        result = client.query("SELECT 42 AS answer")
        assert result["answer"].iloc[0] == 42
    # After __exit__, conn should be None
    assert client._conn is None


def test_close_is_idempotent(tmp_path):
    client = DuckDBClient(db_path=tmp_path / "idem.duckdb")
    client.close()
    client.close()  # Must not raise


def test_conn_property_raises_after_close(tmp_path):
    client = DuckDBClient(db_path=tmp_path / "closed.duckdb")
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = client.conn


# ---------------------------------------------------------------------------
# View registration
# ---------------------------------------------------------------------------


def test_no_views_when_dirs_missing(tmp_path):
    client = DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=tmp_path / "nonexistent_raw",
        canonical_root=tmp_path / "nonexistent_canonical",
    )
    views = client.list_views()
    assert "raw_ohlcv" not in views
    assert "canonical_ohlcv" not in views
    client.close()


def test_raw_view_registered_when_data_present(tmp_path):
    _write_parquet(tmp_path / "raw" / "symbol=AAPL" / "date=2024-01-01", symbol="AAPL")
    client = DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
    )
    assert "raw_ohlcv" in client.list_views()
    client.close()


def test_refresh_views_picks_up_new_data(tmp_path):
    raw_root = tmp_path / "raw"
    client = DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=raw_root,
        canonical_root=tmp_path / "canonical",
    )
    assert "raw_ohlcv" not in client.list_views()

    _write_parquet(raw_root / "symbol=AAPL" / "date=2024-01-01", symbol="AAPL")
    client.refresh_views()
    assert "raw_ohlcv" in client.list_views()
    client.close()


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------


def test_query_returns_dataframe(tmp_path):
    with DuckDBClient(db_path=tmp_path / "q.duckdb") as client:
        df = client.query("SELECT 1 AS x, 'hello' AS y")
    assert isinstance(df, pd.DataFrame)
    assert df["x"].iloc[0] == 1
    assert df["y"].iloc[0] == "hello"


def test_query_with_positional_params(tmp_path):
    with DuckDBClient(db_path=tmp_path / "p.duckdb") as client:
        df = client.query("SELECT ? + ? AS total", params=[3, 4])
    assert df["total"].iloc[0] == 7


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def test_row_count(tmp_path):
    _write_parquet(tmp_path / "raw" / "symbol=AAPL" / "date=2024-01-01", symbol="AAPL", n=5)
    with DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
    ) as client:
        assert client.row_count("raw_ohlcv") == 5


def test_describe_returns_column_metadata(tmp_path):
    _write_parquet(tmp_path / "raw" / "symbol=AAPL" / "date=2024-01-01", symbol="AAPL")
    with DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
    ) as client:
        df = client.describe("raw_ohlcv")
    assert isinstance(df, pd.DataFrame)
    assert "column_name" in df.columns
    assert "close" in df["column_name"].tolist()


def test_symbol_summary_multi_symbol(tmp_path):
    for sym in ["AAPL", "MSFT"]:
        _write_parquet(
            tmp_path / "canonical" / f"symbol={sym}" / "date=2024-01-01",
            symbol=sym,
            n=3,
        )
    with DuckDBClient(
        db_path=tmp_path / "t.duckdb",
        raw_root=tmp_path / "raw",
        canonical_root=tmp_path / "canonical",
    ) as client:
        summary = client.symbol_summary("canonical_ohlcv")

    assert set(summary["symbol"].tolist()) == {"AAPL", "MSFT"}
    assert (summary["row_count"] == 3).all()
