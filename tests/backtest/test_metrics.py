"""Tests for backtest/metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_data_platform.backtest.metrics import (
    MetricsReport,
    annualized_return,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_known_series():
    returns = pd.Series([0.001, 0.002, -0.001, 0.003, -0.002])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert abs(sharpe_ratio(returns) - expected) < 1e-10


def test_sharpe_ratio_positive_for_consistent_gains():
    returns = pd.Series([0.001] * 10 + [-0.0005] * 5)
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_zero_std_positive_mean():
    # Constant positive returns → std=0, mean>0 → Sharpe = +inf
    returns = pd.Series([0.01] * 5)
    result = sharpe_ratio(returns)
    assert math.isinf(result) and result > 0


def test_sharpe_ratio_zero_std_zero_mean():
    returns = pd.Series([0.0] * 5)
    result = sharpe_ratio(returns)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_known_series():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])
    downside = returns[returns < 0]
    expected = returns.mean() / downside.std() * np.sqrt(252)
    assert abs(sortino_ratio(returns) - expected) < 1e-10


def test_sortino_handles_zero_downside():
    # All positive returns → downside vol = 0 → Sortino = +inf
    returns = pd.Series([0.01, 0.02, 0.005, 0.015])
    result = sortino_ratio(returns)
    assert math.isinf(result) and result > 0


def test_sortino_empty_series():
    result = sortino_ratio(pd.Series([], dtype=float))
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_known_sequence():
    # +10% then -20%: drawdown from peak 1.1 → 0.88 = 20%
    returns = pd.Series([0.10, -0.20])
    mdd = max_drawdown(returns)
    np.testing.assert_allclose(mdd, 0.20, rtol=1e-6)


def test_max_drawdown_monotone_increase():
    # No drawdown on a rising series
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_returns_positive():
    returns = pd.Series([0.05, -0.10, 0.02, -0.05, 0.08])
    mdd = max_drawdown(returns)
    assert mdd >= 0.0


def test_max_drawdown_empty():
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------


def test_hit_rate_known_series():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
    assert hit_rate(returns) == pytest.approx(0.6)


def test_hit_rate_all_positive():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert hit_rate(returns) == pytest.approx(1.0)


def test_hit_rate_all_negative():
    returns = pd.Series([-0.01, -0.02])
    assert hit_rate(returns) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


def test_annualized_return_known_series():
    # 252 trading days each returning +0.01% → annualized ≈ e^(252×0.0001) - 1 ≈ 2.74%
    # More precisely: (1.0001)^252 - 1
    daily_r = 0.0001
    returns = pd.Series([daily_r] * 252)
    expected = (1 + daily_r) ** 252 - 1
    result = annualized_return(returns, periods=252)
    assert abs(result - expected) < 1e-10


def test_annualized_return_empty():
    assert annualized_return(pd.Series([], dtype=float)) == 0.0


def test_annualized_return_consistent_sign():
    pos_returns = pd.Series([0.001] * 100)
    neg_returns = pd.Series([-0.001] * 100)
    assert annualized_return(pos_returns) > 0
    assert annualized_return(neg_returns) < 0


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------


def test_turnover_no_change_is_zero():
    # Constant weights → zero daily change
    weights = pd.DataFrame({"AAPL": [0.5, 0.5, 0.5], "MSFT": [-0.5, -0.5, -0.5]})
    assert turnover(weights) == pytest.approx(0.0)


def test_turnover_full_flip():
    # Full flip every day: each weight changes by 1.0 per symbol
    weights = pd.DataFrame(
        {"AAPL": [0.5, -0.5, 0.5], "MSFT": [-0.5, 0.5, -0.5]}
    )
    # diff row 0→1: |(-0.5 - 0.5)| + |(0.5 - (-0.5))| = 2, day-0 diff is NaN→0
    # diff row 1→2: same = 2
    # mean = 2 / 3 rows but first row diff is NaN (fillna 0 in diff)
    # diff row 0→1: |(-0.5 - 0.5)| + |(0.5 - (-0.5))| = 2
    # diff row 1→2: same = 2
    # First row diff is NaN (excluded by min_count=1); mean([2, 2]) = 2.0
    result = turnover(weights)
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# MetricsReport
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_report():
    returns = pd.Series([0.001, -0.0005, 0.002, -0.001, 0.0015])
    report = compute_metrics(returns)
    assert isinstance(report, MetricsReport)
    assert report.turnover is None


def test_compute_metrics_with_weights_df():
    returns = pd.Series([0.001, 0.002, -0.001])
    weights = pd.DataFrame({"AAPL": [0.5, 0.5, 0.3], "MSFT": [-0.5, -0.5, -0.3]})
    report = compute_metrics(returns, weights_df=weights)
    assert report.turnover is not None


def test_compute_metrics_summary_contains_sharpe():
    returns = pd.Series([0.001, -0.0005, 0.002])
    report = compute_metrics(returns)
    assert "Sharpe" in report.summary()
