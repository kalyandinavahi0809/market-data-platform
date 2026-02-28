from pathlib import Path

from market_data_platform.storage.duckdb_client import DuckDBClient


def test_duckdb_client_creates_db_file(tmp_path):
    db_path = tmp_path / "test.duckdb"
    client = DuckDBClient(db_path=db_path)
    assert db_path.exists(), "DuckDB file should be created on init"
    assert client.db_path == db_path
