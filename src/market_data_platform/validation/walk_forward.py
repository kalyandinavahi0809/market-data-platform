"""Rolling walk-forward train/test splitter.

Generates non-overlapping (train, test) window pairs from a panel DataFrame
indexed by trading dates.  No data leakage — test window always starts the
trading day after the last training day.

Usage
-----
>>> splitter = WalkForwardSplitter(train_period=504, test_period=63, step_size=63)
>>> for train_df, test_df in splitter.split(canonical_df):
...     # train: fit / IS eval
...     # test:  OOS eval — zero overlap with train
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Generates rolling train/test splits for walk-forward validation.

    The splitter operates on a sorted list of unique trading dates derived
    from the ``ts_utc`` column (or DatetimeIndex) of the input DataFrame.
    Windows slide forward by *step_size* trading days each iteration.

    Parameters
    ----------
    train_period:
        Number of trading days in each training window.  Default 504 (≈ 2 years).
    test_period:
        Number of trading days in each test (OOS) window.  Default 63 (≈ 1 quarter).
    step_size:
        Number of trading days to advance between consecutive windows.
        Default 63 (non-overlapping quarterly test windows).
    min_train:
        Minimum required training days.  Windows with fewer training days
        than *min_train* are silently skipped.  Default 252 (≈ 1 year).
    """

    def __init__(
        self,
        train_period: int = 504,
        test_period: int = 63,
        step_size: int = 63,
        min_train: int = 252,
    ) -> None:
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.min_train = min_train
        self._splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def split(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split *df* into (train, test) window pairs.

        Parameters
        ----------
        df:
            Panel DataFrame.  Must have ``ts_utc`` as a column **or** a
            ``DatetimeIndex`` named ``ts_utc``.  Multiple rows per date
            (one per symbol) are handled correctly.

        Returns
        -------
        list of (train_df, test_df) tuples
            Every test window starts strictly after the corresponding training
            window ends — no overlap, no leakage.
        """
        dates = self._extract_dates(df)

        if len(dates) == 0:
            self._splits = []
            return []

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        start = 0

        while True:
            train_end_idx = start + self.train_period
            test_end_idx = train_end_idx + self.test_period

            # Not enough dates left to form a full training window
            if train_end_idx > len(dates):
                break

            train_dates = dates[start:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]

            # No test dates available
            if len(test_dates) == 0:
                break

            # Skip if training window is below minimum
            if len(train_dates) < self.min_train:
                start += self.step_size
                continue

            train_df = self._filter(df, train_dates)
            test_df = self._filter(df, test_dates)

            splits.append((train_df, test_df))
            start += self.step_size

        self._splits = splits

        logger.info(
            "WalkForwardSplitter: %d windows (train=%d, test=%d, step=%d)",
            len(splits),
            self.train_period,
            self.test_period,
            self.step_size,
        )
        return splits

    @property
    def n_splits(self) -> int:
        """Number of walk-forward windows from the most recent :meth:`split` call."""
        return len(self._splits)

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _extract_dates(df: pd.DataFrame) -> list:
        """Return a sorted list of unique trading dates from *df*."""
        if "ts_utc" in df.columns:
            return sorted(df["ts_utc"].unique())
        if isinstance(df.index, pd.DatetimeIndex):
            return sorted(df.index.unique())
        raise ValueError(
            "df must have 'ts_utc' as a column or a DatetimeIndex. "
            f"Got columns={list(df.columns)}, index.dtype={df.index.dtype}"
        )

    @staticmethod
    def _filter(df: pd.DataFrame, dates: list) -> pd.DataFrame:
        """Return rows of *df* whose ts_utc (or index) is in *dates*."""
        date_set = set(dates)
        if "ts_utc" in df.columns:
            return df[df["ts_utc"].isin(date_set)].copy()
        return df[df.index.isin(date_set)].copy()
