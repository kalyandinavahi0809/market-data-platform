"""Raw → canonical normalizer.

Reads raw hive-partitioned parquet files, enforces the canonical schema,
deduplicates on (ts_utc, symbol), and returns a clean DataFrame ready for
the canonical writer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from market_data_platform.storage.schema_registry import CANONICAL_OHLCV_SCHEMA

logger = logging.getLogger(__name__)

_CANONICAL_COLUMNS = ["ts_utc", "symbol", "open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast price and volume columns to float64; ts_utc to UTC datetime."""
    for col in _PRICE_COLUMNS + ["volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    return df


def normalize(
    raw_root: Path,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Read raw parquet partitions and return a canonical DataFrame.

    Parameters
    ----------
    raw_root:
        Root directory of the raw hive-partitioned store (e.g. ``data/raw``).
    symbol:
        If provided, only read partitions for this symbol.  Otherwise all
        symbols under *raw_root* are loaded.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV with columns: ts_utc, symbol, open, high, low, close,
        volume.  Deduplicated and sorted by (symbol, ts_utc).

    Raises
    ------
    FileNotFoundError
        If *raw_root* does not exist.
    ValueError
        If no parquet files are found under *raw_root* (or the symbol sub-tree).
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    if symbol:
        search_root = raw_root / f"symbol={symbol}"
        if not search_root.exists():
            raise FileNotFoundError(
                f"No raw data for symbol '{symbol}' at {search_root}"
            )
    else:
        search_root = raw_root

    parquet_files = sorted(search_root.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {search_root}")

    logger.info("Loading %d raw parquet file(s) from %s", len(parquet_files), search_root)

    frames = []
    for path in parquet_files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)

    if not frames:
        raise ValueError(f"All parquet files under {search_root} were unreadable.")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw rows", len(df))

    # Hive partition columns (symbol, date) may duplicate data already in the
    # DataFrame.  Keep only canonical columns that are present.
    present = [c for c in _CANONICAL_COLUMNS if c in df.columns]
    df = df[present].copy()

    df = _coerce_dtypes(df)

    before = len(df)
    df = df.drop_duplicates(subset=["ts_utc", "symbol"])
    dupes = before - len(df)
    if dupes:
        logger.info("Dropped %d duplicate rows", dupes)

    df = df.dropna(subset=["ts_utc", "symbol"])

    df = df.sort_values(["symbol", "ts_utc"]).reset_index(drop=True)

    # Validate against canonical schema.
    try:
        CANONICAL_OHLCV_SCHEMA.schema.validate(df)
    except Exception as exc:
        raise ValueError(f"Canonical schema validation failed: {exc}") from exc

    logger.info("Normalization complete: %d canonical rows", len(df))
    return df
