"""Tests for costs/cost_engine.py."""

from __future__ import annotations

import pandas as pd
import pytest

from market_data_platform.costs.commission import BpsCommission, FixedCommission
from market_data_platform.costs.cost_engine import CostEngine, CostReport
from market_data_platform.costs.slippage import LinearSlippage
from market_data_platform.costs.spread import ConstantSpread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = pd.Timestamp("2024-01-02", tz="UTC")


def _make_engine(slippage_bps=5.0, commission_bps=5.0, spread_bps=5.0):
    return CostEngine(
        slippage_model=LinearSlippage(slippage_bps),
        commission_model=BpsCommission(commission_bps),
        spread_model=ConstantSpread(spread_bps),
    )


def _simple_trades(shares=1000.0, price=100.0):
    return pd.DataFrame(
        {"symbol": ["AAPL"], "date": [_DATE], "shares": [shares], "direction": [1]}
    )


def _simple_prices(price=100.0, vol=0.01, adv=1_000_000):
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [_DATE],
            "close": [price],
            "vol_20d": [vol],
            "adv_20d": [adv],
        }
    )


# ---------------------------------------------------------------------------
# CostReport structure
# ---------------------------------------------------------------------------


def test_cost_report_has_expected_columns():
    engine = _make_engine()
    report = engine.apply(_simple_trades(), _simple_prices())
    expected_cols = {
        "symbol", "date", "shares", "notional",
        "slippage_cost", "commission_cost", "spread_cost", "total_cost",
    }
    assert expected_cols.issubset(set(report.cost_per_trade.columns))


# ---------------------------------------------------------------------------
# Known-input total_cost_bps
# ---------------------------------------------------------------------------


def test_total_cost_bps_known_input():
    """1000 shares × $100 = $100k notional; 5+5+5 = 15 bps → $150 / $100k × 10000 = 15 bps."""
    engine = _make_engine(slippage_bps=5.0, commission_bps=5.0, spread_bps=5.0)
    report = engine.apply(_simple_trades(1000, 100.0), _simple_prices(100.0))
    assert report.total_cost_bps == pytest.approx(15.0)
    assert report.total_cost_dollars == pytest.approx(150.0)


def test_total_cost_dollars_matches_sum_of_components():
    engine = _make_engine()
    report = engine.apply(_simple_trades(), _simple_prices())
    row_sum = (
        report.cost_per_trade["slippage_cost"]
        + report.cost_per_trade["commission_cost"]
        + report.cost_per_trade["spread_cost"]
    ).sum()
    assert report.total_cost_dollars == pytest.approx(row_sum)


# ---------------------------------------------------------------------------
# cost_attribution sums to total
# ---------------------------------------------------------------------------


def test_cost_attribution_sums_to_total():
    engine = _make_engine()
    report = engine.apply(_simple_trades(), _simple_prices())
    attributed_total = sum(report.cost_attribution.values())
    assert attributed_total == pytest.approx(report.total_cost_dollars)


def test_cost_attribution_has_all_keys():
    engine = _make_engine()
    report = engine.apply(_simple_trades(), _simple_prices())
    assert set(report.cost_attribution.keys()) == {"slippage", "commission", "spread"}


# ---------------------------------------------------------------------------
# Zero trades → zero cost
# ---------------------------------------------------------------------------


def test_zero_trades_produces_zero_cost():
    engine = _make_engine()
    empty_trades = pd.DataFrame(
        columns=["symbol", "date", "shares", "direction"]
    )
    report = engine.apply(empty_trades, _simple_prices())
    assert report.total_cost_dollars == pytest.approx(0.0)
    assert report.total_cost_bps == pytest.approx(0.0)
    assert report.cost_per_trade.empty


def test_zero_trades_net_returns_equals_gross():
    engine = _make_engine()
    gross = pd.Series([0.001, 0.002, -0.001], index=pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC"))
    empty_trades = pd.DataFrame(columns=["symbol", "date", "shares", "direction"])
    report = engine.apply(empty_trades, _simple_prices(), gross_returns=gross)
    pd.testing.assert_series_equal(report.net_returns, gross, check_names=False)


# ---------------------------------------------------------------------------
# net_returns < gross_returns
# ---------------------------------------------------------------------------


def test_net_returns_less_than_gross_always():
    engine = _make_engine()
    dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    trades = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": dates,
            "shares": [1_000.0, 1_000.0, 1_000.0],
            "direction": [1, 1, 1],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": dates,
            "close": [100.0, 101.0, 100.5],
            "vol_20d": [0.01, 0.01, 0.01],
            "adv_20d": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    gross = pd.Series([0.005, 0.003, 0.004], index=dates)
    report = engine.apply(trades, prices, gross_returns=gross, portfolio_value=1_000_000.0)
    assert (report.net_returns < report.gross_returns).all()


# ---------------------------------------------------------------------------
# Missing symbols handled gracefully
# ---------------------------------------------------------------------------


def test_missing_symbol_in_prices_produces_zero_cost():
    """Symbol in trades_df missing from prices_df → no crash, zero cost."""
    engine = _make_engine()
    trades = pd.DataFrame(
        {"symbol": ["UNKNOWN"], "date": [_DATE], "shares": [500.0], "direction": [1]}
    )
    prices = _simple_prices()  # only AAPL
    report = engine.apply(trades, prices)
    assert report.total_cost_dollars == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Composable models
# ---------------------------------------------------------------------------


def test_composable_models_fixed_commission():
    """CostEngine works with FixedCommission."""
    engine = CostEngine(
        slippage_model=LinearSlippage(5.0),
        commission_model=FixedCommission(per_trade=10.0),
        spread_model=ConstantSpread(5.0),
    )
    report = engine.apply(_simple_trades(1000, 100.0), _simple_prices(100.0))
    # slippage = $50, commission = $10, spread = $50 → total = $110
    assert report.total_cost_dollars == pytest.approx(110.0)
    assert report.cost_attribution["commission"] == pytest.approx(10.0)
