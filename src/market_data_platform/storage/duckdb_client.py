"""DuckDB client for the market-data-platform.

Lightweight connection manager and query interface.

- Persistent DuckDB database file at data/market_data.duckdb by default
- Registers raw and canonical parquet partitions as views (if present)
- Supports context manager protocol
- Exposes utility methods: list_views, row_count, describe, symbol_summary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

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

        self._conn: Optional[duckdb.DuckDBPyConnection] = duckdb.connect(
            str(self.db_path), read_only=read_only
        )
        logger.debug(
            "Connected to DuckDB at %s (read_only=%s)", self.db_path, read_only
        )

        self._register_view("raw_ohlcv", self.raw_root)
        self._register_view("canonical_ohlcv", self.canonical_root)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DuckDBClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("DuckDB connection closed.")

    # ------------------------------------------------------------------
    # Connection property
    # ------------------------------------------------------------------

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDB connection is closed.")
        return self._conn

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------

    def _register_view(self, view_name: str, root: Path) -> None:
        root = Path(root)
        if not root.exists():
            logger.debug(
                "Skipping view '%s' — directory not found: %s", view_name, root
            )
            return

        parquet_files = list(root.rglob("*.parquet"))
        if not parquet_files:
            logger.debug(
                "Skipping view '%s' — no parquet files under %s", view_name, root
            )
            return

        sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT * FROM read_parquet(
            '{root}/**/*.parquet',
            hive_partitioning = true,
            hive_types_autocast = true
        )
        """
        self.conn.execute(sql)
        logger.info(
            "Registered view '%s' from %s (%d files)",
            view_name,
            root,
            len(parquet_files),
        )

    def refresh_views(self) -> None:
        """Re-register raw and canonical views (call after new data lands)."""
        self._register_view("raw_ohlcv", self.raw_root)
        self._register_view("canonical_ohlcv", self.canonical_root)
        logger.debug("Views refreshed.")

    def list_views(self) -> List[str]:
        """Return names of all views registered in this DuckDB connection."""
        df = self.conn.execute(
            "SELECT view_name FROM duckdb_views() ORDER BY view_name"
        ).df()
        views = df["view_name"].tolist()
        logger.debug("list_views → %s", views)
        return views

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute *sql* and return results as a DataFrame.

        Parameters
        ----------
        sql:
            SQL string, optionally with positional ``?`` placeholders.
        params:
            List of values to bind to ``?`` placeholders, or *None*.
        """
        result = self.conn.execute(sql, params or [])
        return result.df()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def row_count(self, view_or_table: str) -> int:
        """Return the number of rows in *view_or_table*."""
        df = self.conn.execute(f"SELECT COUNT(*) AS n FROM {view_or_table}").df()
        count = int(df["n"].iloc[0])
        logger.debug("row_count(%s) → %d", view_or_table, count)
        return count

    def describe(self, view_or_table: str) -> pd.DataFrame:
        """Return column metadata for *view_or_table*."""
        df = self.conn.execute(f"DESCRIBE {view_or_table}").df()
        logger.debug("describe(%s) → %d columns", view_or_table, len(df))
        return df

    def symbol_summary(self, view_or_table: str = "canonical_ohlcv") -> pd.DataFrame:
        """Return per-symbol row counts and date range from *view_or_table*.

        Expects the view to contain ``symbol`` and ``ts_utc`` columns.
        """
        sql = f"""
        SELECT
            symbol,
            COUNT(*)          AS row_count,
            MIN(ts_utc)::DATE AS min_date,
            MAX(ts_utc)::DATE AS max_date
        FROM {view_or_table}
        GROUP BY symbol
        ORDER BY symbol
        """
        df = self.conn.execute(sql).df()
        logger.debug("symbol_summary(%s) → %d symbols", view_or_table, len(df))
        return df
