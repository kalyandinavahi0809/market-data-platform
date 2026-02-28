"""Idempotent canonical layer writer.

Validates a canonical DataFrame with Pandera, then writes hive-partitioned
parquet files to the canonical store.  Existing partitions are overwritten so
the operation is safe to re-run (idempotent).

Partition scheme:
    data/canonical/symbol=<SYMBOL>/date=<YYYY-MM-DD>/part-<YYYY-MM-DD>.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from market_data_platform.storage.schema_registry import CANONICAL_OHLCV_SCHEMA

logger = logging.getLogger(__name__)

DEFAULT_CANONICAL_ROOT = Path("data/canonical")


def write_canonical(
    df: pd.DataFrame,
    canonical_root: Path = DEFAULT_CANONICAL_ROOT,
) -> int:
    """Validate *df* and write it to the canonical hive-partitioned store.

    Parameters
    ----------
    df:
        A DataFrame conforming to the canonical OHLCV schema.  Must contain
        ``ts_utc``, ``symbol``, ``open``, ``high``, ``low``, ``close``,
        ``volume`` columns.
    canonical_root:
        Root directory for the canonical store.

    Returns
    -------
    int
        Number of partition files written.

    Raises
    ------
    ValueError
        If Pandera schema validation fails.
    """
    if df.empty:
        logger.warning("write_canonical called with empty DataFrame — nothing written.")
        return 0

    # Validate before touching the filesystem.
    try:
        CANONICAL_OHLCV_SCHEMA.schema.validate(df)
    except Exception as exc:
        raise ValueError(f"Canonical schema validation failed: {exc}") from exc

    canonical_root = Path(canonical_root)
    df = df.copy()
    df["_date"] = df["ts_utc"].dt.date

    files_written = 0
    for (symbol, date_val), group in df.groupby(["symbol", "_date"]):
        out_dir = canonical_root / f"symbol={symbol}" / f"date={date_val}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"part-{date_val}.parquet"

        partition = group.drop(columns=["_date"])
        partition.to_parquet(out_path, index=False)
        files_written += 1
        logger.debug("Wrote %d rows → %s", len(partition), out_path)

    logger.info(
        "write_canonical: %d rows → %d partition(s) under %s",
        len(df),
        files_written,
        canonical_root,
    )
    return files_written
