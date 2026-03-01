"""Market-impact / slippage models.

Each model exposes a single method::

    apply(trade_size, price, vol, adv) -> float | array

*trade_size*  – signed or unsigned share quantity (absolute value is used).
*price*       – execution price per share (dollars).
*vol*         – realised volatility of the instrument (fraction, e.g. 0.01 = 1 %).
*adv*         – average daily volume in shares.

All models use ``numpy`` operations so they accept both scalars and
``pandas.Series`` / ``numpy`` arrays without modification.
"""

from __future__ import annotations

import numpy as np


class LinearSlippage:
    """Flat slippage expressed as a fixed number of basis points.

    Parameters
    ----------
    slippage_bps:
        Slippage cost as a fraction of notional, measured in basis points.
        Default 5 bps (0.05 %).
    """

    def __init__(self, slippage_bps: float = 5.0) -> None:
        self.slippage_bps = slippage_bps

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return slippage cost in dollars.

        cost = |trade_size| × price × slippage_bps / 10_000
        """
        notional = abs(trade_size) * price
        return notional * self.slippage_bps / 10_000


class VolatilitySlippage:
    """Volatility-scaled slippage proportional to participation rate.

    Models the intuition that a larger fraction of daily volume → more
    adverse price movement.

    Parameters
    ----------
    k:
        Scaling coefficient.  Default 0.1.
    """

    def __init__(self, k: float = 0.1) -> None:
        self.k = k

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return slippage cost in dollars.

        participation_rate = |trade_size| / adv  (0 when adv = 0)
        cost = notional × k × vol × sqrt(participation_rate)
        """
        notional = abs(trade_size) * price
        # Use adv_safe to prevent ZeroDivisionError before np.where masks the result
        adv_safe = np.where(adv != 0, adv, 1.0)
        participation_rate = np.where(adv != 0, abs(trade_size) / adv_safe, 0.0)
        return notional * self.k * vol * np.sqrt(participation_rate)


class SquareRootImpact:
    """Almgren-style square-root market-impact model.

    Approximates the price impact of large trades as proportional to the
    square root of the fraction of ADV being traded.

    Parameters
    ----------
    sigma_coeff:
        Volatility coefficient.  Default 0.1.
    """

    def __init__(self, sigma_coeff: float = 0.1) -> None:
        self.sigma_coeff = sigma_coeff

    def apply(self, trade_size, price, vol, adv) -> float:
        """Return market-impact cost in dollars.

        adv_fraction = |trade_size| / adv  (0 when adv = 0)
        cost = notional × sigma_coeff × vol × sqrt(adv_fraction)
        """
        notional = abs(trade_size) * price
        adv_safe = np.where(adv != 0, adv, 1.0)
        adv_fraction = np.where(adv != 0, abs(trade_size) / adv_safe, 0.0)
        return notional * self.sigma_coeff * vol * np.sqrt(adv_fraction)
