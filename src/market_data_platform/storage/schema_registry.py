"""Schema registry for market_data_platform.

This module defines schema contracts for datasets used by the platform.

Note: The goal of this repository is production-style architecture; keep this
module small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandera as pa


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    schema: pa.DataFrameSchema


RAW_OHLCV_SCHEMA = DatasetSchema(
    name="raw_ohlcv",
    schema=pa.DataFrameSchema(
        {
            "ts_utc": pa.Column(pa.DateTime, nullable=False),
            "symbol": pa.Column(str, nullable=False),
            "open": pa.Column(float, nullable=False),
            "high": pa.Column(float, nullable=False),
            "low": pa.Column(float, nullable=False),
            "close": pa.Column(float, nullable=False),
            "volume": pa.Column(float, nullable=False),
            "ingested_at_utc": pa.Column(pa.DateTime, nullable=False),
            "source": pa.Column(str, nullable=False),
        },
        strict=False,
    ),
)


CANONICAL_OHLCV_SCHEMA = DatasetSchema(
    name="canonical_ohlcv",
    schema=pa.DataFrameSchema(
        {
            "ts_utc": pa.Column(pa.DateTime, nullable=False),
            "symbol": pa.Column(str, nullable=False),
            "open": pa.Column(float, nullable=False),
            "high": pa.Column(float, nullable=False),
            "low": pa.Column(float, nullable=False),
            "close": pa.Column(float, nullable=False),
            "volume": pa.Column(float, nullable=False),
        },
        strict=False,
    ),
)
