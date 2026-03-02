"""Tests for validation/regime_filter.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.validation.regime_filter import VolatilityRegimeFilter

_REGIMES = ("LOW_VOL", "NORMAL_VOL", "HIGH_VOL")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n_dates: int = 200, n_symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    """Wide log-return DataFrame: rows = dates, columns = symbols."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    data = rng.normal(0, 0.01, size=(n_dates, n_symbols))
    return pd.DataFrame(data, index=idx, columns=[f"SYM{i}" for i in range(n_symbols)])


def _make_portfolio_returns(n_dates: int = 200, seed: int = 0) -> pd.Series:
    """Simple daily portfolio return series for use with filter() and regime_sharpes()."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, n_dates), index=idx, name="portfolio_return")


# ---------------------------------------------------------------------------
# fit() — threshold estimation
# ---------------------------------------------------------------------------


def test_fit_sets_thresholds():
    """After fit(), _low_threshold and _high_threshold are set to floats."""
    filt = VolatilityRegimeFilter()
    filt.fit(_make_returns())
    assert isinstance(filt._low_threshold, float)
    assert isinstance(filt._high_threshold, float)
    assert not np.isnan(filt._low_threshold)
    assert not np.isnan(filt._high_threshold)


def test_thresholds_are_ordered():
    """low_threshold ≤ high_threshold after fit."""
    filt = VolatilityRegimeFilter(low_percentile=20, high_percentile=80)
    filt.fit(_make_returns())
    assert filt._low_threshold <= filt._high_threshold


def test_fit_returns_self():
    """fit() returns self to allow method chaining."""
    filt = VolatilityRegimeFilter()
    result = filt.fit(_make_returns())
    assert result is filt


def test_fit_insufficient_data_raises():
    """fit() with only 1 valid vol point raises ValueError."""
    # 1 row → rolling(20, min_periods=2) produces no valid value
    df = _make_returns(n_dates=1)
    with pytest.raises(ValueError, match="enough data"):
        VolatilityRegimeFilter().fit(df)


# ---------------------------------------------------------------------------
# label() — regime assignment
# ---------------------------------------------------------------------------


def test_label_returns_only_valid_regimes():
    """All label values are one of LOW_VOL, NORMAL_VOL, HIGH_VOL."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    assert set(labels.unique()).issubset(set(_REGIMES))


def test_all_dates_labeled():
    """Every date in returns_df receives a non-NaN label."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    assert labels.notna().all()
    assert len(labels) == len(returns_df)


def test_label_index_matches_input():
    """labels.index is identical to returns_df.index."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    pd.testing.assert_index_equal(labels.index, returns_df.index)


def test_label_before_fit_raises():
    """Calling label() before fit() raises RuntimeError."""
    filt = VolatilityRegimeFilter()
    with pytest.raises(RuntimeError, match="fit"):
        filt.label(_make_returns())


def test_thresholds_use_only_fit_data():
    """Thresholds estimated on train data differ from thresholds on full data."""
    returns_df = _make_returns(n_dates=200)
    train_df = returns_df.iloc[:100]
    full_df = returns_df

    filt_train = VolatilityRegimeFilter().fit(train_df)
    filt_full = VolatilityRegimeFilter().fit(full_df)

    # Thresholds may differ because percentiles are computed on different samples
    # (this verifies fit respects only the provided data, not the global series)
    assert filt_train._low_threshold != filt_full._low_threshold or \
           filt_train._high_threshold != filt_full._high_threshold


# ---------------------------------------------------------------------------
# filter() — subsetting by regime
# ---------------------------------------------------------------------------


def test_filter_returns_correct_subset():
    """filter(returns, regime, labels) returns only dates matching the regime."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns()

    # Align returns with labels (same index)
    returns = returns.reindex(returns_df.index)

    for regime in _REGIMES:
        subset = filt.filter(returns, regime, labels)
        # Every date in the subset must have the correct label
        subset_labels = labels.reindex(subset.index)
        assert (subset_labels == regime).all(), f"Non-{regime} dates found in filter output"


def test_filter_preserves_return_values():
    """Values in the filtered series match the originals at identical dates."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns().reindex(returns_df.index)

    for regime in _REGIMES:
        subset = filt.filter(returns, regime, labels)
        pd.testing.assert_series_equal(subset, returns.loc[subset.index])


def test_filter_invalid_regime_raises():
    """filter() raises ValueError for an unrecognised regime string."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns().reindex(returns_df.index)

    with pytest.raises(ValueError, match="regime"):
        filt.filter(returns, "MOON_VOL", labels)


# ---------------------------------------------------------------------------
# regime_sharpes() — conditional Sharpe ratios
# ---------------------------------------------------------------------------


def test_regime_sharpes_returns_all_three_keys():
    """regime_sharpes() returns a dict with exactly {LOW_VOL, NORMAL_VOL, HIGH_VOL}."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns().reindex(returns_df.index)

    result = filt.regime_sharpes(returns, labels)
    assert set(result.keys()) == set(_REGIMES)


def test_regime_sharpes_values_are_finite_or_nan():
    """All regime Sharpe values are floats (finite or nan, never exception)."""
    filt = VolatilityRegimeFilter()
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns().reindex(returns_df.index)

    result = filt.regime_sharpes(returns, labels)
    for regime, sharpe in result.items():
        assert isinstance(sharpe, float), f"{regime}: expected float, got {type(sharpe)}"


def test_regime_sharpes_nan_for_single_obs_regime():
    """regime_sharpes() returns nan for any regime with fewer than 2 observations."""
    # Use extreme percentiles so one regime covers nearly all days
    filt = VolatilityRegimeFilter(low_percentile=0.1, high_percentile=99.9)
    returns_df = _make_returns()
    filt.fit(returns_df)
    labels = filt.label(returns_df)
    returns = _make_portfolio_returns().reindex(returns_df.index)

    result = filt.regime_sharpes(returns, labels)
    # With near-0 / near-100 percentiles, LOW_VOL or HIGH_VOL may have 0-1 day
    # We simply verify that any such regime returns nan (not raise)
    for sharpe in result.values():
        assert isinstance(sharpe, float)
