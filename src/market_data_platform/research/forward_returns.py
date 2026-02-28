"""Forward return computation for signal validation.

Forward returns represent future price changes and are used **exclusively**
for out-of-sample signal validation and backtesting — never as inputs to
feature computation.

Critical design rule
--------------------
``add_forward_returns`` must always be called *after* ``compute_features``.
The two functions are deliberately separate to make lookahead impossible:
features look backward, forward returns look forward.  Tests explicitly
verify the identity ``fwd_return_1d[t] == log_return_1d[t+1]``.

Forward returns produced
------------------------
fwd_return_1d : ln(close_{t+1} / close_t)  — NaN at last row of each symbol
fwd_return_5d : ln(close_{t+5} / close_t)  — NaN at last 5 rows of each symbol
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append forward return columns to *df*.

    Parameters
    ----------
    df:
        Feature DataFrame (output of ``compute_features``).  Must contain
        ``ts_utc``, ``symbol``, and ``close`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``fwd_return_1d`` and ``fwd_return_5d`` columns
        appended.  Existing feature columns are unchanged.  The last row(s)
        of each symbol will have NaN forward returns (no future data).
    """
    df = df.sort_values(["symbol", "ts_utc"]).reset_index(drop=True).copy()

    df["_log_close"] = np.log(df["close"])

    g = df.groupby("symbol", sort=False)

    # fwd_return_1d[t] = log(close[t+1]) - log(close[t])
    # Equivalently: -diff(-1) within each symbol group
    df["fwd_return_1d"] = g["_log_close"].transform(lambda s: s.shift(-1) - s)
    df["fwd_return_5d"] = g["_log_close"].transform(lambda s: s.shift(-5) - s)

    df = df.drop(columns=["_log_close"])

    logger.debug(
        "add_forward_returns: %d rows, fwd_return_1d NaN count=%d",
        len(df),
        df["fwd_return_1d"].isna().sum(),
    )
    return df
