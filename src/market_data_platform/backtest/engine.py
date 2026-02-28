"""Vectorized long-short equity portfolio simulator.

Strategy
--------
Signal  : cross-sectional momentum rank (``cs_rank`` from ``log_return_20d``)
Long    : top quintile  — symbols with cs_rank > +0.6
Short   : bottom quintile — symbols with cs_rank < −0.6
Weights : equal-weight within each side, dollar neutral (gross long = 1,
          gross short = 1, net = 0)
Rebalance: daily
Costs   : none in Phase 2 (added in Phase 3)

All computation is vectorized — no iterrows.  The engine accepts a fully
prepared DataFrame (features + cross-section + forward returns merged) and
returns a ``pd.Series`` of daily portfolio returns indexed by ``ts_utc``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LONG_THRESHOLD: float = 0.6
SHORT_THRESHOLD: float = -0.6


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "cs_rank",
    return_col: str = "fwd_return_1d",
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.Series:
    """Simulate the cross-sectional momentum strategy.

    Parameters
    ----------
    df:
        Prepared DataFrame with at least ``ts_utc``, ``symbol``,
        *signal_col*, and *return_col* columns.  Typically the output of::

            df = compute_features(canonical_df)
            df = compute_cross_section(df)
            df = add_forward_returns(df)

    signal_col:
        Column used to determine long/short positions.
    return_col:
        Column used to compute realised returns (should be forward-looking).
    long_threshold:
        Minimum *signal_col* value to be included in the long book.
    short_threshold:
        Maximum *signal_col* value to be included in the short book.

    Returns
    -------
    pd.Series
        Daily portfolio returns indexed by ``ts_utc``, named
        ``"portfolio_return"``.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Assign raw positions: +1 long, −1 short, 0 flat                    #
    # ------------------------------------------------------------------ #
    df["_position"] = 0.0
    df.loc[df[signal_col] > long_threshold, "_position"] = 1.0
    df.loc[df[signal_col] < short_threshold, "_position"] = -1.0

    # ------------------------------------------------------------------ #
    # Count longs and shorts per date (vectorized, no loops)             #
    # ------------------------------------------------------------------ #
    long_count = df.groupby("ts_utc")["_position"].transform(
        lambda s: (s > 0).sum()
    )
    short_count = df.groupby("ts_utc")["_position"].transform(
        lambda s: (s < 0).sum()
    )

    # ------------------------------------------------------------------ #
    # Equal weight within each side                                       #
    # Gross long = 1, gross short = 1  →  net = 0 (dollar neutral)      #
    # ------------------------------------------------------------------ #
    df["_weight"] = 0.0

    long_mask = df["_position"] > 0
    short_mask = df["_position"] < 0

    # Replace 0 counts with NaN to avoid division by zero, then fill 0
    df.loc[long_mask, "_weight"] = (
        1.0 / long_count[long_mask].replace(0, np.nan)
    ).fillna(0.0)
    df.loc[short_mask, "_weight"] = (
        -1.0 / short_count[short_mask].replace(0, np.nan)
    ).fillna(0.0)

    # ------------------------------------------------------------------ #
    # Portfolio return = Σ weight_i × return_i                           #
    # NaN forward returns (last day) contribute 0 to avoid distortion   #
    # ------------------------------------------------------------------ #
    df["_contribution"] = df["_weight"] * df[return_col].fillna(0.0)
    portfolio_returns = df.groupby("ts_utc")["_contribution"].sum()
    portfolio_returns.name = "portfolio_return"

    logger.info(
        "run_backtest: %d trading days, mean daily return=%.4f%%",
        len(portfolio_returns),
        portfolio_returns.mean() * 100,
    )
    return portfolio_returns
