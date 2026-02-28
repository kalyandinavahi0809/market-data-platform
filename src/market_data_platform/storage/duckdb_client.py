"""DuckDB client for the market-data-platform.

This is a lightweight connection manager and query interface.

- Uses a persistent DuckDB database file at data/market_data.duckdb
- Registers raw and canonical parquet partitions as views (if present)
- Exposes a .query() method returning a pandas DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("data/market_data.duckdb")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_CANONICAL_ROOT = Path("data/canonical")


class DuckDBClient:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        raw_root: Path = DEFAULT_RAW_ROOT,
        canonical_root: Path = DEFAULT_CANONICAL_ROOT,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.raw_root = Path(raw_root)
        self.canonical_root = Path(canonical_root)
        self.read_only = read_only

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[duckdb.DuckDBPyConnection]
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)

        # Register views; if dirs are missing, skip silently.
        self._register_view("raw_ohlcv", self.raw_root)
        self._register_view("canonical_ohlcv", self.canonical_root)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def _register_view(self, view_name: str, root: Path) -> None:
        root = Path(root)
        if not root.exists():
            return

        sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT * FROM read_parquet(
            '{root}/**/*.parquet',
            hive_partitioning = true,
            hive_types_autocast = true
        )
        """
        self._conn.execute(sql)

    def refresh_views(self) -> None:
        self._register_view("raw_ohlcv", self.raw_root)
        self._register_view("canonical_ohlcv", self.canonical_root)

    def query(self, sql: str, **params) -> pd.DataFrame:
        return self._conn.execute(sql, params).df()
