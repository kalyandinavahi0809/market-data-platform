"""Bid-ask spread cost models.

Each model exposes::

    apply(trade_size, price, vol, adv) -> float | array

Costs represent the *half-spread* per transaction — the full round-trip
cost is ``2 × half_spread`` (paid on entry and again on exit).

All models accept both scalars and ``pandas.Series`` / ``numpy`` arrays.
"""

from __future__ import annotations

import numpy as np


class ConstantSpread:
    """Fixed half-spread cost as a constant number of basis points.

    Parameters
    ----------
    half_spread_bps:
        Half of the bid-ask spread in basis points.  Default 5 bps.
        The full round-trip cost is ``2 × half_spread_bps`` bps per
        unit of notional.
    """

    def __init__(self, half_spread_bps: float = 5.0) -> None:
        self.half_spread_bps = half_spread_bps

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return half-spread cost in dollars.

        cost = |trade_size| × price × half_spread_bps / 10_000
        """
        notional = abs(trade_size) * price
        return notional * self.half_spread_bps / 10_000


class VolatilitySpread:
    """Volatility-adaptive half-spread model.

    Wider spreads during high-volatility regimes, subject to a minimum floor.

    Parameters
    ----------
    k:
        Scaling coefficient applied to ``vol × 10_000`` to derive the
        half-spread in bps.  Default 0.5.
    min_bps:
        Minimum half-spread floor in basis points.  Default 2.0 bps.
    """

    def __init__(self, k: float = 0.5, min_bps: float = 2.0) -> None:
        self.k = k
        self.min_bps = min_bps

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return half-spread cost in dollars.

        half_spread_bps = max(min_bps, k × vol × 10_000)
        cost = notional × half_spread_bps / 10_000
        """
        notional = abs(trade_size) * price
        half_spread_bps = np.maximum(self.min_bps, self.k * vol * 10_000)
        return notional * half_spread_bps / 10_000
