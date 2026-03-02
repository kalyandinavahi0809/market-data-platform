"""Tests for validation/oos_evaluator.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.validation.oos_evaluator import OOSEvaluator, OOSReport, WindowResult
from market_data_platform.validation.walk_forward import WalkForwardSplitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPLITTER_KWARGS = dict(train_period=30, test_period=10, step_size=10, min_train=25)


def _make_canonical(n_dates: int = 60, n_symbols: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic canonical OHLCV DataFrame suitable for the full pipeline.

    Symbols have slightly different drift to produce varied cross-sectional signals.
    Using log-price random walk so ``close`` is always positive.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B", tz="UTC")
    rows = []
    for sym_i in range(n_symbols):
        sym = f"SYM{sym_i}"
        log_prices = np.cumsum(rng.normal(0.001 * (sym_i + 1 - n_symbols / 2), 0.02, n_dates))
        prices = 100.0 * np.exp(log_prices)
        for j, dt in enumerate(dates):
            p = prices[j]
            rows.append(
                {
                    "ts_utc": dt,
                    "symbol": sym,
                    "open": p * 0.99,
                    "high": p * 1.01,
                    "low": p * 0.98,
                    "close": p,
                    "volume": float(rng.integers(500_000, 2_000_000)),
                }
            )
    return pd.DataFrame(rows)


def _run_default(n_dates: int = 60, n_symbols: int = 10) -> tuple:
    """Return (evaluator, report) for a default-config run."""
    df = _make_canonical(n_dates=n_dates, n_symbols=n_symbols)
    splitter = WalkForwardSplitter(**_SPLITTER_KWARGS)
    evaluator = OOSEvaluator(splitter=splitter)
    report = evaluator.run(df)
    return evaluator, report


# ---------------------------------------------------------------------------
# Structure tests (don't touch exact numeric values)
# ---------------------------------------------------------------------------


def test_oos_report_has_correct_n_windows():
    """OOSReport.n_windows matches expected number of walk-forward splits."""
    _, report = _run_default(n_dates=60)
    assert report.n_windows == 3


def test_window_results_count_matches_n_windows():
    """len(window_results) tuple equals n_windows."""
    _, report = _run_default(n_dates=60)
    assert len(report.window_results) == report.n_windows


def test_oos_returns_is_series_with_name():
    """OOSReport.oos_returns is a pd.Series named 'oos_portfolio_return'."""
    _, report = _run_default(n_dates=60)
    assert isinstance(report.oos_returns, pd.Series)
    assert report.oos_returns.name == "oos_portfolio_return"


def test_oos_returns_not_empty():
    """Stitched OOS returns contain at least one non-NaN value."""
    _, report = _run_default(n_dates=60)
    assert report.oos_returns.dropna().shape[0] > 0


def test_window_result_is_frozen_dataclass():
    """WindowResult is an immutable frozen dataclass."""
    _, report = _run_default(n_dates=60)
    wr = report.window_results[0]
    with pytest.raises((AttributeError, TypeError)):
        wr.window_idx = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Date ordering and leakage
# ---------------------------------------------------------------------------


def test_all_window_results_have_correct_date_order():
    """For each window: train_start ≤ train_end < test_start ≤ test_end."""
    _, report = _run_default(n_dates=60)
    for wr in report.window_results:
        assert wr.train_start <= wr.train_end
        assert wr.train_end < wr.test_start
        assert wr.test_start <= wr.test_end


def test_consecutive_windows_non_overlapping_test_dates():
    """Test windows from consecutive walk-forward splits do not overlap."""
    _, report = _run_default(n_dates=60)
    all_test_starts = [wr.test_start for wr in report.window_results]
    all_test_ends = [wr.test_end for wr in report.window_results]
    for i in range(1, len(all_test_starts)):
        # Next window's test start must be after previous window's test end
        assert all_test_starts[i] > all_test_ends[i - 1]


# ---------------------------------------------------------------------------
# Aggregate metric consistency
# ---------------------------------------------------------------------------


def test_is_sharpe_average_of_window_sharpes():
    """report.is_sharpe equals the nanmean of per-window IS Sharpes."""
    _, report = _run_default(n_dates=60)
    expected = float(np.nanmean([wr.is_sharpe for wr in report.window_results]))
    assert report.is_sharpe == pytest.approx(expected, abs=1e-10)


def test_sharpe_degradation_formula_matches():
    """sharpe_degradation_pct == (is_sharpe - oos_sharpe) / |is_sharpe| * 100."""
    _, report = _run_default(n_dates=60)
    is_sh = report.is_sharpe
    oos_sh = report.oos_sharpe
    # Skip check when degenerate (near-zero IS Sharpe or NaN)
    if np.isnan(is_sh) or np.isnan(oos_sh) or abs(is_sh) < 1e-10:
        pytest.skip("Degenerate Sharpe values — cannot verify degradation formula")
    expected = (is_sh - oos_sh) / abs(is_sh) * 100.0
    assert report.sharpe_degradation_pct == pytest.approx(expected, rel=1e-6)


def test_oos_net_sharpe_is_nan_without_cost_engine():
    """Without a cost engine, oos_net_sharpe must be nan."""
    _, report = _run_default(n_dates=60)
    assert np.isnan(report.oos_net_sharpe)


# ---------------------------------------------------------------------------
# summary() method
# ---------------------------------------------------------------------------


def test_summary_contains_key_labels():
    """summary() output contains expected section labels."""
    evaluator, _ = _run_default(n_dates=60)
    s = evaluator.summary()
    assert "OOS Sharpe" in s
    assert "IS Sharpe" in s
    assert "Windows" in s


def test_summary_before_run_returns_placeholder():
    """Calling summary() before run() returns an informative placeholder string."""
    splitter = WalkForwardSplitter(**_SPLITTER_KWARGS)
    evaluator = OOSEvaluator(splitter=splitter)
    s = evaluator.summary()
    assert "not yet run" in s.lower() or "call run" in s.lower()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_run_raises_on_zero_windows():
    """run() raises ValueError when the splitter produces zero windows."""
    # 20 dates < train_period=30 → no windows
    df = _make_canonical(n_dates=20)
    splitter = WalkForwardSplitter(**_SPLITTER_KWARGS)
    evaluator = OOSEvaluator(splitter=splitter)
    with pytest.raises(ValueError, match="zero windows"):
        evaluator.run(df)
