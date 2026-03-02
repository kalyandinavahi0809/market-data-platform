"""Volatility regime classification for conditional performance analysis.

Classifies each trading day into LOW_VOL, NORMAL_VOL, or HIGH_VOL based on
the rolling realised volatility of an equal-weight universe portfolio.
Percentile thresholds are estimated on TRAINING data only; test data is then
labelled using those fixed thresholds.

Typical usage
-------------
>>> filt = VolatilityRegimeFilter(low_percentile=20, high_percentile=80)
>>> filt.fit(train_returns_df)
>>> labels = filt.label(oos_returns_df)
>>> sharpes = filt.regime_sharpes(oos_portfolio_returns, labels)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from market_data_platform.backtest.metrics import sharpe_ratio as _sharpe

logger = logging.getLogger(__name__)

_REGIMES = ("LOW_VOL", "NORMAL_VOL", "HIGH_VOL")
_ROLL = 20  # days used for rolling vol
_ANN = np.sqrt(252)


class VolatilityRegimeFilter:
    """Classify trading days by the volatility regime of the equal-weight portfolio.

    Regimes are defined by percentile thresholds of the rolling 20-day
    realised volatility, estimated on **training data only**.

    Parameters
    ----------
    low_percentile:
        Days at or below this percentile are labelled ``LOW_VOL``.  Default 20.
    high_percentile:
        Days above this percentile are labelled ``HIGH_VOL``.  Default 80.
    """

    def __init__(
        self,
        low_percentile: float = 20.0,
        high_percentile: float = 80.0,
    ) -> None:
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self._low_threshold: float | None = None
        self._high_threshold: float | None = None

    # ---------------------------------------------------------------------- #
    # Fitting                                                                  #
    # ---------------------------------------------------------------------- #

    def fit(self, returns_df: pd.DataFrame) -> "VolatilityRegimeFilter":
        """Estimate vol percentile thresholds from *returns_df*.

        Parameters
        ----------
        returns_df:
            Wide DataFrame — columns are symbol names, index is dates,
            values are log returns.  Must be sorted by date ascending.

        Returns
        -------
        self
            Allows method chaining: ``filt.fit(train_df).label(test_df)``.
        """
        vol = self._rolling_vol(returns_df)
        valid = vol.dropna().values

        if len(valid) < 2:
            raise ValueError(
                f"Not enough data to estimate vol percentiles. Got {len(valid)} valid values."
            )

        self._low_threshold = float(np.percentile(valid, self.low_percentile))
        self._high_threshold = float(np.percentile(valid, self.high_percentile))

        logger.info(
            "VolatilityRegimeFilter fitted: low_thresh=%.4f, high_thresh=%.4f",
            self._low_threshold,
            self._high_threshold,
        )
        return self

    # ---------------------------------------------------------------------- #
    # Labelling                                                                #
    # ---------------------------------------------------------------------- #

    def label(self, returns_df: pd.DataFrame) -> pd.Series:
        """Assign a regime label to each date in *returns_df*.

        Parameters
        ----------
        returns_df:
            Wide DataFrame of log returns (same schema as used in ``fit``).

        Returns
        -------
        pd.Series
            Index matches *returns_df*.index; values in
            ``{'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL'}``.  All dates receive
            a label — dates where rolling vol is undefined (start of series)
            default to ``'NORMAL_VOL'``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()

        vol = self._rolling_vol(returns_df)

        # Default: NORMAL_VOL for all dates (handles NaN vol at series start)
        labels = pd.Series("NORMAL_VOL", index=returns_df.index, dtype=object)
        valid = vol.notna()
        labels[valid & (vol <= self._low_threshold)] = "LOW_VOL"
        labels[valid & (vol > self._high_threshold)] = "HIGH_VOL"

        return labels

    # ---------------------------------------------------------------------- #
    # Filtering and conditional metrics                                        #
    # ---------------------------------------------------------------------- #

    def filter(
        self,
        returns: pd.Series,
        regime: str,
        labels: pd.Series,
    ) -> pd.Series:
        """Return only the *returns* that fall in the specified *regime*.

        Parameters
        ----------
        returns:
            Daily portfolio return series.
        regime:
            One of ``'LOW_VOL'``, ``'NORMAL_VOL'``, ``'HIGH_VOL'``.
        labels:
            Regime label series from :meth:`label` — index must be
            compatible with *returns*.

        Returns
        -------
        pd.Series
            Subset of *returns* where ``labels == regime``.
        """
        if regime not in _REGIMES:
            raise ValueError(f"regime must be one of {_REGIMES}; got '{regime}'")
        mask = labels == regime
        # Align on common index
        aligned = mask.reindex(returns.index).fillna(False)
        return returns[aligned]

    def regime_sharpes(
        self,
        returns: pd.Series,
        labels: pd.Series,
        periods: int = 252,
    ) -> Dict[str, float]:
        """Compute annualised Sharpe ratio for each volatility regime.

        Parameters
        ----------
        returns:
            Daily portfolio return series.
        labels:
            Regime label series from :meth:`label`.
        periods:
            Trading periods per year.  Default 252.

        Returns
        -------
        dict
            ``{'LOW_VOL': float, 'NORMAL_VOL': float, 'HIGH_VOL': float}``.
            ``nan`` for any regime with fewer than 2 observations.
        """
        result: Dict[str, float] = {}
        for regime in _REGIMES:
            subset = self.filter(returns, regime, labels)
            if len(subset) < 2:
                result[regime] = float("nan")
            else:
                result[regime] = _sharpe(subset, periods)
        return result

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _rolling_vol(returns_df: pd.DataFrame) -> pd.Series:
        """Equal-weight portfolio rolling 20-day annualised vol."""
        eq_ret = returns_df.mean(axis=1)
        return eq_ret.rolling(_ROLL, min_periods=2).std() * _ANN

    def _check_fitted(self) -> None:
        if self._low_threshold is None:
            raise RuntimeError(
                "VolatilityRegimeFilter has not been fitted. "
                "Call fit(returns_df) before label() or regime_sharpes()."
            )
