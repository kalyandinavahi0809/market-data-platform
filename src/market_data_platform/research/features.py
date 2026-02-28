"""Per-symbol time-series feature engineering.

Computes momentum, volatility, and volume features from canonical OHLCV data.
All operations are fully vectorized using pandas groupby + transform; no
iterrows are used anywhere in this module.

Point-in-time safety guarantee
--------------------------------
Every feature at time *t* is computed exclusively from data at *t* and
earlier observations for the same symbol.  Adding future rows to the input
DataFrame will never change feature values at any existing timestamp.

Features produced
-----------------
log_return_1d  : ln(close_t / close_{t-1})
log_return_5d  : ln(close_t / close_{t-5})
log_return_20d : ln(close_t / close_{t-20})
vol_20d        : 20-day rolling std of log_return_1d, annualized (×√252)
volume_zscore  : (volume_t - 20d_mean) / 20d_std
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "vol_20d",
    "volume_zscore",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-series features for every symbol in *df*.

    Parameters
    ----------
    df:
        Canonical OHLCV DataFrame.  Must contain ``ts_utc``, ``symbol``,
        ``close``, and ``volume`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional feature columns appended.  Sorted by
        ``(symbol, ts_utc)``.  NaNs appear at the start of each symbol's
        history where window requirements are not yet met.
    """
    df = df.sort_values(["symbol", "ts_utc"]).reset_index(drop=True).copy()

    df["_log_close"] = np.log(df["close"])

    g = df.groupby("symbol", sort=False)

    # ------------------------------------------------------------------ #
    # Log returns — vectorized grouped diff, no cross-symbol contamination #
    # ------------------------------------------------------------------ #
    df["log_return_1d"] = g["_log_close"].diff(1)
    df["log_return_5d"] = g["_log_close"].diff(5)
    df["log_return_20d"] = g["_log_close"].diff(20)

    # ------------------------------------------------------------------ #
    # 20-day annualized volatility of daily log returns                   #
    # min_periods=20 ensures NaN for the first 19 return observations     #
    # ------------------------------------------------------------------ #
    df["vol_20d"] = g["log_return_1d"].transform(
        lambda s: s.rolling(20, min_periods=20).std() * np.sqrt(252)
    )

    # ------------------------------------------------------------------ #
    # Volume z-score: (volume - 20d mean) / 20d std                      #
    # ------------------------------------------------------------------ #
    roll_mean = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    roll_std = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["volume_zscore"] = (df["volume"] - roll_mean) / roll_std

    df = df.drop(columns=["_log_close"])

    logger.debug(
        "compute_features: %d rows, %d symbols",
        len(df),
        df["symbol"].nunique(),
    )
    return df
