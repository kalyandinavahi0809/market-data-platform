"""Commission models.

Each model exposes::

    apply(trade_size, price, vol, adv) -> float | array

All dollar amounts are in USD; all rates are in basis points (bps).
All models accept both scalars and ``pandas.Series`` / ``numpy`` arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FixedCommission:
    """Flat per-trade commission, independent of size.

    Parameters
    ----------
    per_trade:
        Fixed cost per trade execution in dollars.  Default $1.00.
    """

    def __init__(self, per_trade: float = 1.0) -> None:
        self.per_trade = per_trade

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return flat commission in dollars.

        cost = per_trade  (regardless of size)
        """
        notional = abs(trade_size) * price
        return np.where(notional != 0, self.per_trade, 0.0)


class BpsCommission:
    """Commission as a fixed number of basis points of notional.

    Parameters
    ----------
    bps:
        Commission rate in basis points.  Default 5 bps.
    """

    def __init__(self, bps: float = 5.0) -> None:
        self.bps = bps

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return commission in dollars.

        cost = |trade_size| × price × bps / 10_000
        """
        notional = abs(trade_size) * price
        return notional * self.bps / 10_000


class TieredCommission:
    """Volume-discount commission with notional-based tiers.

    The tier with the *lowest* rate for which ``notional >= threshold`` is
    applied.  When no tier threshold is satisfied, the highest rate (worst
    tier) is used as a fallback.

    Parameters
    ----------
    tiers:
        List of ``(notional_threshold, bps_rate)`` tuples.
        Example: ``[(0, 10.0), (10_000, 7.0), (100_000, 5.0)]``
    """

    def __init__(self, tiers: list[tuple[float, float]]) -> None:
        self.tiers = tiers

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return commission in dollars using the best eligible tier.

        Among all tiers where ``notional >= threshold``, the lowest rate
        (best volume discount) is applied.  Falls back to the highest rate
        when notional is below all thresholds.
        """
        notional = abs(trade_size) * price

        if not self.tiers:
            return notional * 0.0

        max_rate = max(r for _, r in self.tiers)

        is_scalar = np.isscalar(notional)
        n_arr = np.atleast_1d(np.asarray(notional, dtype=float))

        # Start with fallback: the highest rate (applies when below all thresholds)
        rates = np.full(len(n_arr), max_rate, dtype=float)

        # For each tier, wherever the tier is eligible AND improves the rate, apply it
        for threshold, rate in self.tiers:
            eligible = n_arr >= threshold
            better = rate < rates
            rates = np.where(eligible & better, rate, rates)

        result = n_arr * rates / 10_000

        if is_scalar:
            return float(result[0])
        index = getattr(notional, "index", None)
        return pd.Series(result, index=index)
