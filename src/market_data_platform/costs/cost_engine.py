"""Composable transaction cost engine.

``CostEngine`` wires together a slippage model, a commission model, and a
spread model and applies them to a trades log to produce a ``CostReport``.

Usage::

    engine = CostEngine(
        slippage_model=LinearSlippage(5.0),
        commission_model=BpsCommission(5.0),
        spread_model=ConstantSpread(5.0),
    )
    report = engine.apply(trades_df, prices_df, gross_returns=gross_series)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CostReport:
    """Aggregated transaction cost results.

    Attributes
    ----------
    cost_per_trade:
        Per-trade cost breakdown with columns:
        symbol, date, shares, notional, slippage_cost, commission_cost,
        spread_cost, total_cost.
    total_cost_dollars:
        Sum of all per-trade costs in dollars.
    total_cost_bps:
        ``total_cost_dollars / total_notional × 10_000``.
    gross_returns:
        Daily gross portfolio return series (``None`` if not provided).
    net_returns:
        Gross returns minus daily cost drag (``None`` if not provided).
    cost_attribution:
        Dict with keys ``"slippage"``, ``"commission"``, ``"spread"`` and
        their respective total dollar amounts.
    """

    cost_per_trade: pd.DataFrame
    total_cost_dollars: float
    total_cost_bps: float
    gross_returns: Optional[pd.Series]
    net_returns: Optional[pd.Series]
    cost_attribution: dict


class CostEngine:
    """Composable engine that applies slippage, commission, and spread costs.

    Parameters
    ----------
    slippage_model:
        Instance of any slippage model exposing ``apply(...)``.
    commission_model:
        Instance of any commission model exposing ``apply(...)``.
    spread_model:
        Instance of any spread model exposing ``apply(...)``.
    """

    def __init__(self, slippage_model, commission_model, spread_model) -> None:
        self.slippage_model = slippage_model
        self.commission_model = commission_model
        self.spread_model = spread_model

    def apply(
        self,
        trades_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        gross_returns: Optional[pd.Series] = None,
        portfolio_value: float = 1_000_000.0,
    ) -> CostReport:
        """Compute per-trade costs and aggregate into a ``CostReport``.

        Parameters
        ----------
        trades_df:
            Columns: ``symbol``, ``date``, ``shares`` (absolute quantity),
            ``direction`` (+1 long, -1 short).
        prices_df:
            Columns: ``symbol``, ``date``, ``close``, ``vol_20d``,
            ``adv_20d``.  Missing symbols are handled gracefully (zero cost).
        gross_returns:
            Optional daily gross return series indexed by date.  When
            provided, ``net_returns`` is computed as
            ``gross - daily_cost_drag``.
        portfolio_value:
            Dollar value of the portfolio used to convert dollar costs to
            return fractions.  Default $1 000 000.

        Returns
        -------
        CostReport
        """
        # ------------------------------------------------------------------ #
        # Empty trades fast-path                                               #
        # ------------------------------------------------------------------ #
        if trades_df.empty:
            empty = pd.DataFrame(
                columns=[
                    "symbol", "date", "shares", "notional",
                    "slippage_cost", "commission_cost", "spread_cost",
                    "total_cost",
                ]
            )
            return CostReport(
                cost_per_trade=empty,
                total_cost_dollars=0.0,
                total_cost_bps=0.0,
                gross_returns=gross_returns,
                net_returns=gross_returns,
                cost_attribution={"slippage": 0.0, "commission": 0.0, "spread": 0.0},
            )

        # ------------------------------------------------------------------ #
        # Merge trades with prices (left join — missing symbols → NaN)        #
        # ------------------------------------------------------------------ #
        prices_clean = prices_df[
            ["symbol", "date", "close", "vol_20d", "adv_20d"]
        ].copy() if {"vol_20d", "adv_20d"}.issubset(prices_df.columns) else _ensure_price_cols(prices_df)

        merged = trades_df[["symbol", "date", "shares", "direction"]].merge(
            prices_clean, on=["symbol", "date"], how="left"
        )

        # Fill missing price-related fields to make cost = 0 for unknown symbols
        merged["close"] = merged["close"].fillna(0.0)
        merged["vol_20d"] = merged["vol_20d"].fillna(0.0)
        merged["adv_20d"] = merged["adv_20d"].fillna(0.0)

        # ------------------------------------------------------------------ #
        # Compute notional (vectorized)                                        #
        # ------------------------------------------------------------------ #
        merged["notional"] = merged["shares"].abs() * merged["close"]

        # ------------------------------------------------------------------ #
        # Apply each cost model (each model supports Series inputs)           #
        # ------------------------------------------------------------------ #
        slippage_costs = pd.Series(
            np.asarray(
                self.slippage_model.apply(
                    merged["shares"],
                    merged["close"],
                    merged["vol_20d"],
                    merged["adv_20d"],
                ),
                dtype=float,
            ),
            index=merged.index,
        ).fillna(0.0)

        commission_costs = pd.Series(
            np.asarray(
                self.commission_model.apply(
                    merged["shares"],
                    merged["close"],
                    merged["vol_20d"],
                    merged["adv_20d"],
                ),
                dtype=float,
            ),
            index=merged.index,
        ).fillna(0.0)

        spread_costs = pd.Series(
            np.asarray(
                self.spread_model.apply(
                    merged["shares"],
                    merged["close"],
                    merged["vol_20d"],
                    merged["adv_20d"],
                ),
                dtype=float,
            ),
            index=merged.index,
        ).fillna(0.0)

        merged["slippage_cost"] = slippage_costs
        merged["commission_cost"] = commission_costs
        merged["spread_cost"] = spread_costs
        merged["total_cost"] = (
            merged["slippage_cost"]
            + merged["commission_cost"]
            + merged["spread_cost"]
        )

        # ------------------------------------------------------------------ #
        # Aggregate totals                                                     #
        # ------------------------------------------------------------------ #
        total_cost_dollars = float(merged["total_cost"].sum())
        total_notional = float(merged["notional"].sum())
        total_cost_bps = (
            total_cost_dollars / total_notional * 10_000
            if total_notional > 0
            else 0.0
        )

        cost_attribution = {
            "slippage": float(merged["slippage_cost"].sum()),
            "commission": float(merged["commission_cost"].sum()),
            "spread": float(merged["spread_cost"].sum()),
        }

        # ------------------------------------------------------------------ #
        # Net returns: gross minus daily cost drag                            #
        # ------------------------------------------------------------------ #
        net_returns: Optional[pd.Series] = None
        if gross_returns is not None:
            daily_cost = (
                merged.groupby("date")["total_cost"]
                .sum()
                .reindex(gross_returns.index, fill_value=0.0)
            )
            # Convert dollar cost to return fraction
            daily_cost_return = daily_cost / portfolio_value
            net_returns = gross_returns - daily_cost_return
            net_returns.name = "net_portfolio_return"

        cost_per_trade = merged[
            [
                "symbol", "date", "shares", "notional",
                "slippage_cost", "commission_cost", "spread_cost",
                "total_cost",
            ]
        ].reset_index(drop=True)

        logger.info(
            "CostEngine: %d trades, total_cost=$%.2f, total_cost_bps=%.2f",
            len(merged),
            total_cost_dollars,
            total_cost_bps,
        )

        return CostReport(
            cost_per_trade=cost_per_trade,
            total_cost_dollars=total_cost_dollars,
            total_cost_bps=total_cost_bps,
            gross_returns=gross_returns,
            net_returns=net_returns,
            cost_attribution=cost_attribution,
        )


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #


def _ensure_price_cols(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add missing vol_20d and adv_20d columns (defaults to 0)."""
    df = prices_df.copy()
    if "vol_20d" not in df.columns:
        df["vol_20d"] = 0.0
    if "adv_20d" not in df.columns:
        df["adv_20d"] = 0.0
    return df[["symbol", "date", "close", "vol_20d", "adv_20d"]]
