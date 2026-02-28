"""Schema registry for market_data_platform.

Defines Pandera schema contracts for all dataset layers: raw, canonical,
and research features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import pandera.pandas as pa

# pandas 2.0+ returns timezone-aware datetime64[us, UTC] from parquet.
# coerce=True on each schema lets Pandera convert between compatible datetime
# representations (ns vs us resolution) without failing validation.
_TS_DTYPE = pd.DatetimeTZDtype(tz="UTC")


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    schema: pa.DataFrameSchema


RAW_OHLCV_SCHEMA = DatasetSchema(
    name="raw_ohlcv",
    schema=pa.DataFrameSchema(
        {
            "ts_utc": pa.Column(_TS_DTYPE, nullable=False, coerce=True),
            "symbol": pa.Column(str, nullable=False),
            "open": pa.Column(float, nullable=False, coerce=True),
            "high": pa.Column(float, nullable=False, coerce=True),
            "low": pa.Column(float, nullable=False, coerce=True),
            "close": pa.Column(float, nullable=False, coerce=True),
            "volume": pa.Column(float, nullable=False, coerce=True),
            "ingested_at_utc": pa.Column(_TS_DTYPE, nullable=False, coerce=True),
            "source": pa.Column(str, nullable=False),
        },
        strict=False,
    ),
)

CANONICAL_OHLCV_SCHEMA = DatasetSchema(
    name="canonical_ohlcv",
    schema=pa.DataFrameSchema(
        {
            "ts_utc": pa.Column(_TS_DTYPE, nullable=False, coerce=True),
            "symbol": pa.Column(str, nullable=False),
            "open": pa.Column(float, nullable=False, coerce=True),
            "high": pa.Column(float, nullable=False, coerce=True),
            "low": pa.Column(float, nullable=False, coerce=True),
            "close": pa.Column(float, nullable=False, coerce=True),
            "volume": pa.Column(float, nullable=False, coerce=True),
        },
        strict=False,
    ),
)

RESEARCH_FEATURES_SCHEMA = DatasetSchema(
    name="research_features",
    schema=pa.DataFrameSchema(
        {
            "ts_utc": pa.Column(_TS_DTYPE, nullable=False, coerce=True),
            "symbol": pa.Column(str, nullable=False),
            "close": pa.Column(float, nullable=False, coerce=True),
            "ret_1d": pa.Column(float, nullable=True, coerce=True),
            "ret_5d": pa.Column(float, nullable=True, coerce=True),
            "ret_20d": pa.Column(float, nullable=True, coerce=True),
            "vol_20d": pa.Column(float, nullable=True, coerce=True),
            "volume_zscore": pa.Column(float, nullable=True, coerce=True),
        },
        strict=False,
    ),
)

_SCHEMA_REGISTRY: Dict[str, DatasetSchema] = {
    "raw": RAW_OHLCV_SCHEMA,
    "canonical": CANONICAL_OHLCV_SCHEMA,
    "research": RESEARCH_FEATURES_SCHEMA,
}


def get_schema(layer: str) -> DatasetSchema:
    """Return the DatasetSchema for *layer*.

    Parameters
    ----------
    layer:
        One of ``"raw"``, ``"canonical"``, or ``"research"``.

    Raises
    ------
    KeyError
        If *layer* is not registered.
    """
    try:
        return _SCHEMA_REGISTRY[layer]
    except KeyError:
        valid = ", ".join(sorted(_SCHEMA_REGISTRY))
        raise KeyError(f"Unknown layer '{layer}'. Valid layers: {valid}")
