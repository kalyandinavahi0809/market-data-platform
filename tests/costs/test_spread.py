"""Tests for costs/spread.py."""

from __future__ import annotations

import pytest

from market_data_platform.costs.spread import ConstantSpread, VolatilitySpread


# ---------------------------------------------------------------------------
# ConstantSpread
# ---------------------------------------------------------------------------


def test_constant_spread_known_input():
    """1000 shares × $100 × 5 bps half-spread = $50."""
    model = ConstantSpread(half_spread_bps=5.0)
    assert model.apply(1000, 100.0, 0.01, 100_000) == pytest.approx(50.0)


def test_constant_spread_scales_with_notional():
    model = ConstantSpread(half_spread_bps=5.0)
    c1 = model.apply(100, 10.0, 0.01, 0)
    c2 = model.apply(1000, 10.0, 0.01, 0)
    assert c2 == pytest.approx(10 * c1)


def test_constant_spread_zero_for_zero_trade():
    model = ConstantSpread(half_spread_bps=5.0)
    assert model.apply(0, 50.0, 0.01, 0) == pytest.approx(0.0)


def test_constant_spread_default_bps():
    """Default half_spread_bps = 5.0 bps."""
    model = ConstantSpread()
    # 200 shares × $100 = $20k, 5 bps → $10
    assert model.apply(200, 100.0, 0.0, 0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# VolatilitySpread
# ---------------------------------------------------------------------------


def test_volatility_spread_widens_with_vol():
    model = VolatilitySpread(k=0.5, min_bps=2.0)
    low_vol_cost = model.apply(1000, 100.0, vol=0.001, adv=0)
    high_vol_cost = model.apply(1000, 100.0, vol=0.10, adv=0)
    assert high_vol_cost > low_vol_cost


def test_volatility_spread_respects_min_bps_floor():
    """When k × vol × 10_000 < min_bps, cost = notional × min_bps / 10_000."""
    model = VolatilitySpread(k=0.5, min_bps=3.0)
    # vol so small that k * vol * 10000 = 0.5 * 0.0001 * 10000 = 0.5 bps < 3.0 bps floor
    cost = model.apply(1000, 50.0, vol=0.0001, adv=0)
    expected = 1000 * 50.0 * 3.0 / 10_000  # floor applies
    assert cost == pytest.approx(expected)


def test_volatility_spread_above_floor():
    """When k × vol × 10_000 > min_bps, the vol term dominates."""
    model = VolatilitySpread(k=1.0, min_bps=2.0)
    # vol = 0.05 → vol*10000 = 500 bps >> 2 bps floor
    cost = model.apply(100, 10.0, vol=0.05, adv=0)
    expected = 100 * 10.0 * (1.0 * 0.05 * 10_000) / 10_000
    assert cost == pytest.approx(expected)


def test_volatility_spread_zero_for_zero_trade():
    model = VolatilitySpread(k=0.5, min_bps=2.0)
    assert model.apply(0, 100.0, vol=0.05, adv=0) == pytest.approx(0.0)


def test_constant_spread_non_negative():
    model = ConstantSpread(half_spread_bps=5.0)
    assert model.apply(1000, 100.0, 0.0, 0) >= 0.0


def test_volatility_spread_non_negative():
    model = VolatilitySpread(k=0.5, min_bps=2.0)
    assert model.apply(1000, 100.0, 0.02, 1_000_000) >= 0.0
