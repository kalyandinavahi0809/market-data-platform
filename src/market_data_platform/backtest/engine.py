"""Vectorized long-short equity portfolio simulator.

Strategy
--------
Signal  : cross-sectional momentum rank (``cs_rank`` from ``log_return_20d``)
Long    : top quintile  — symbols with cs_rank > +0.6
Short   : bottom quintile — symbols with cs_rank < −0.6
Weights : equal-weight within each side, dollar neutral (gross long = 1,
          gross short = 1, net = 0)
Rebalance: daily

Functions
---------
run_backtest      : gross returns (no costs)
run_with_costs    : gross + net returns via a CostEngine

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


def _compute_weights(
    df: pd.DataFrame,
    signal_col: str = "cs_rank",
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.DataFrame:
    """Assign portfolio weights from a cross-sectional signal.

    Returns a copy of *df* with ``_position`` and ``_weight`` columns added.
    Internal helper shared by :func:`run_backtest` and :func:`run_with_costs`.
    """
    df = df.copy()

    df["_position"] = 0.0
    df.loc[df[signal_col] > long_threshold, "_position"] = 1.0
    df.loc[df[signal_col] < short_threshold, "_position"] = -1.0

    long_count = df.groupby("ts_utc")["_position"].transform(
        lambda s: (s > 0).sum()
    )
    short_count = df.groupby("ts_utc")["_position"].transform(
        lambda s: (s < 0).sum()
    )

    df["_weight"] = 0.0
    long_mask = df["_position"] > 0
    short_mask = df["_position"] < 0

    df.loc[long_mask, "_weight"] = (
        1.0 / long_count[long_mask].replace(0, np.nan)
    ).fillna(0.0)
    df.loc[short_mask, "_weight"] = (
        -1.0 / short_count[short_mask].replace(0, np.nan)
    ).fillna(0.0)

    return df


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "cs_rank",
    return_col: str = "fwd_return_1d",
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.Series:
    """Simulate the cross-sectional momentum strategy (gross returns).

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
    df_w = _compute_weights(df, signal_col, long_threshold, short_threshold)

    # ------------------------------------------------------------------ #
    # Portfolio return = Σ weight_i × return_i                           #
    # NaN forward returns (last day) contribute 0 to avoid distortion   #
    # ------------------------------------------------------------------ #
    df_w["_contribution"] = df_w["_weight"] * df_w[return_col].fillna(0.0)
    portfolio_returns = df_w.groupby("ts_utc")["_contribution"].sum()
    portfolio_returns.name = "portfolio_return"

    logger.info(
        "run_backtest: %d trading days, mean daily return=%.4f%%",
        len(portfolio_returns),
        portfolio_returns.mean() * 100,
    )
    return portfolio_returns


def run_with_costs(
    df: pd.DataFrame,
    cost_engine,  # CostEngine (avoid circular import — accept duck-typed)
    signal_col: str = "cs_rank",
    return_col: str = "fwd_return_1d",
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
    portfolio_value: float = 1_000_000.0,
):
    """Run the backtest and apply transaction costs via *cost_engine*.

    Derives a trades log from daily weight changes (full daily rebalance),
    passes it through *cost_engine*, and returns a ``CostReport`` containing
    both gross and net portfolio returns.

    Parameters
    ----------
    df:
        Prepared DataFrame (same as :func:`run_backtest`).
    cost_engine:
        A ``CostEngine`` instance.
    portfolio_value:
        Dollar size of the portfolio used to convert weight changes to share
        quantities.  Default $1 000 000.

    Returns
    -------
    CostReport
        Contains ``gross_returns``, ``net_returns``, per-trade cost breakdown,
        and attribution.
    """
    # ------------------------------------------------------------------ #
    # 1. Gross returns                                                    #
    # ------------------------------------------------------------------ #
    gross_returns = run_backtest(df, signal_col, return_col, long_threshold, short_threshold)

    # ------------------------------------------------------------------ #
    # 2. Daily weights (wide format: index=ts_utc, columns=symbol)       #
    # ------------------------------------------------------------------ #
    df_w = _compute_weights(df, signal_col, long_threshold, short_threshold)
    weights_wide = df_w.pivot_table(
        index="ts_utc", columns="symbol", values="_weight", fill_value=0.0
    )

    # ------------------------------------------------------------------ #
    # 3. Daily weight changes → trades                                   #
    # First day enters at the initial weights (no prior portfolio).      #
    # ------------------------------------------------------------------ #
    weight_changes = weights_wide.diff()
    weight_changes.iloc[0] = weights_wide.iloc[0]

    # Stack to long format
    wc = weight_changes.stack(future_stack=True)
    wc.name = "weight_change"
    trades = wc.reset_index()
    trades.columns = ["date", "symbol", "weight_change"]

    # Keep only actual rebalance trades
    trades = trades[trades["weight_change"].abs() > 1e-10].copy()

    # Convert weight changes to shares using closing prices
    close_map = (
        df[["ts_utc", "symbol", "close"]]
        .rename(columns={"ts_utc": "date"})
        .drop_duplicates(subset=["date", "symbol"])
    )
    trades = trades.merge(close_map, on=["date", "symbol"], how="left")
    # Avoid division by zero for missing prices
    close_safe = trades["close"].fillna(1.0).replace(0.0, 1.0)
    trades["shares"] = trades["weight_change"].abs() * portfolio_value / close_safe
    trades["direction"] = np.sign(trades["weight_change"]).astype(int)
    trades_df = trades[["symbol", "date", "shares", "direction"]].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 4. Prices DataFrame: close, vol_20d, adv_20d                       #
    # ------------------------------------------------------------------ #
    prices_df = (
        df[["ts_utc", "symbol", "close"]]
        .rename(columns={"ts_utc": "date"})
        .drop_duplicates(subset=["date", "symbol"])
        .copy()
    )
    if "vol_20d" in df.columns:
        vol_map = (
            df[["ts_utc", "symbol", "vol_20d"]]
            .rename(columns={"ts_utc": "date"})
            .drop_duplicates(subset=["date", "symbol"])
        )
        prices_df = prices_df.merge(vol_map, on=["date", "symbol"], how="left")
    else:
        prices_df["vol_20d"] = 0.0

    adv_20d = df.groupby("symbol", sort=False)["volume"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )
    adv_df = (
        df[["ts_utc", "symbol"]]
        .rename(columns={"ts_utc": "date"})
        .assign(adv_20d=adv_20d.values)
        .drop_duplicates(subset=["date", "symbol"])
    )
    prices_df = prices_df.merge(adv_df, on=["date", "symbol"], how="left")

    # ------------------------------------------------------------------ #
    # 5. Apply cost engine                                                #
    # ------------------------------------------------------------------ #
    return cost_engine.apply(
        trades_df,
        prices_df,
        gross_returns=gross_returns,
        portfolio_value=portfolio_value,
    )
