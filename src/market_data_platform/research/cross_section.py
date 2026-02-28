"""Cross-sectional signal normalization.

At each timestamp, symbols are ranked by a chosen signal and the ranks are
normalized to a common scale.  This removes the time-series level of the
signal (e.g. overall market trends) and isolates the *relative* information.

Normalization schemes
---------------------
cs_rank   : rank / (n - 1) × 2 − 1, mapped to [−1, 1]
            −1 = weakest signal, 0 = median, +1 = strongest signal
cs_zscore : (signal − cross_mean) / cross_std at each timestamp

Edge cases
----------
* Symbols with NaN signal at a timestamp are excluded from that date's
  cross-section and receive NaN cs_rank / cs_zscore.
* When only one valid symbol exists at a timestamp, that symbol receives
  cs_rank = 0 (neutral) and cs_zscore = 0.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cross_section(
    df: pd.DataFrame,
    signal_col: str = "log_return_20d",
) -> pd.DataFrame:
    """Add cross-sectional rank and z-score columns to *df*.

    Parameters
    ----------
    df:
        Features DataFrame.  Must contain ``ts_utc``, ``symbol``, and
        *signal_col*.
    signal_col:
        Column to rank cross-sectionally.  Default: ``"log_return_20d"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``cs_rank`` and ``cs_zscore`` columns appended.
    """
    df = df.copy()

    # Number of valid (non-NaN) signals per timestamp
    n_valid = df.groupby("ts_utc")[signal_col].transform(
        lambda s: s.notna().sum()
    )

    # 0-indexed rank within each timestamp; NaN signals receive NaN rank
    raw_rank = (
        df.groupby("ts_utc")[signal_col].rank(method="average", na_option="keep") - 1
    )

    # Normalize: rank / (n − 1) × 2 − 1  →  [−1, 1]
    # Clamp denominator to 1 to avoid ZeroDivisionError when n ≤ 1
    denom = (n_valid - 1).clip(lower=1)
    df["cs_rank"] = raw_rank / denom * 2 - 1

    # Special case: single valid symbol → neutral rank (0)
    single_valid = (n_valid == 1) & df[signal_col].notna()
    df.loc[single_valid, "cs_rank"] = 0.0

    # Cross-sectional z-score
    def _zscore(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 2:
            # Return 0 for the one valid symbol; NaN for NaN signals
            return s.where(s.isna(), 0.0)
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0:
            return s.where(s.isna(), 0.0)
        return (s - mu) / sigma

    df["cs_zscore"] = df.groupby("ts_utc")[signal_col].transform(_zscore)

    logger.debug(
        "compute_cross_section: signal=%s, valid_rank_count=%d",
        signal_col,
        df["cs_rank"].notna().sum(),
    )
    return df
