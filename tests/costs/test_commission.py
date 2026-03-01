"""Tests for costs/commission.py."""

from __future__ import annotations

import pandas as pd
import pytest

from market_data_platform.costs.commission import (
    BpsCommission,
    FixedCommission,
    TieredCommission,
)


# ---------------------------------------------------------------------------
# FixedCommission
# ---------------------------------------------------------------------------


def test_fixed_commission_is_flat_regardless_of_size():
    model = FixedCommission(per_trade=2.50)
    cost_small = model.apply(10, 5.0, 0.01, 100_000)
    cost_large = model.apply(100_000, 5000.0, 0.01, 100_000)
    assert cost_small == pytest.approx(2.50)
    assert cost_large == pytest.approx(2.50)


def test_fixed_commission_zero_for_zero_trade():
    model = FixedCommission(per_trade=1.0)
    assert model.apply(0, 100.0, 0.01, 100_000) == pytest.approx(0.0)


def test_fixed_commission_default_is_one_dollar():
    model = FixedCommission()
    assert model.apply(500, 50.0, 0.0, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BpsCommission
# ---------------------------------------------------------------------------


def test_bps_commission_known_input():
    """1000 shares × $100 × 5 bps = $50."""
    model = BpsCommission(bps=5.0)
    assert model.apply(1000, 100.0, 0.01, 0) == pytest.approx(50.0)


def test_bps_commission_scales_linearly_with_notional():
    model = BpsCommission(bps=3.0)
    c1 = model.apply(100, 10.0, 0.01, 0)    # notional = $1k
    c2 = model.apply(200, 10.0, 0.01, 0)    # notional = $2k
    assert c2 == pytest.approx(2 * c1)


def test_bps_commission_zero_for_zero_trade():
    model = BpsCommission(bps=10.0)
    assert model.apply(0, 50.0, 0.01, 0) == pytest.approx(0.0)


def test_bps_commission_series_input():
    model = BpsCommission(bps=5.0)
    shares = pd.Series([100.0, 500.0])
    prices = pd.Series([10.0, 20.0])
    result = model.apply(shares, prices, 0.01, 0)
    expected = shares * prices * 5.0 / 10_000
    pd.testing.assert_series_equal(pd.Series(result), expected, check_names=False)


# ---------------------------------------------------------------------------
# TieredCommission
# ---------------------------------------------------------------------------

_TIERS = [(0.0, 10.0), (10_000.0, 7.0), (100_000.0, 5.0)]


def test_tiered_commission_highest_rate_for_small_notional():
    """Below all non-zero thresholds → fallback to highest rate."""
    model = TieredCommission(tiers=[(5_000.0, 10.0), (50_000.0, 7.0)])
    # notional = 100 × $20 = $2k < $5k → max_rate = 10 bps
    cost = model.apply(100, 20.0, 0.01, 0)
    assert cost == pytest.approx(2_000 * 10.0 / 10_000)


def test_tiered_commission_correct_tier_mid():
    """Notional in the middle tier."""
    model = TieredCommission(tiers=_TIERS)
    # 500 shares × $50 = $25k → tier (10_000, 7.0) applies
    cost = model.apply(500, 50.0, 0.01, 0)
    assert cost == pytest.approx(25_000 * 7.0 / 10_000)


def test_tiered_commission_correct_tier_top():
    """Notional above highest threshold → lowest rate."""
    model = TieredCommission(tiers=_TIERS)
    # 2_000 shares × $100 = $200k → tier (100_000, 5.0) applies
    cost = model.apply(2_000, 100.0, 0.01, 0)
    assert cost == pytest.approx(200_000 * 5.0 / 10_000)


def test_tiered_commission_bottom_tier():
    """Notional in the first tier (>= 0 threshold)."""
    model = TieredCommission(tiers=_TIERS)
    # 10 shares × $50 = $500 → tier (0.0, 10.0) applies
    cost = model.apply(10, 50.0, 0.01, 0)
    assert cost == pytest.approx(500 * 10.0 / 10_000)


def test_tiered_commission_empty_tiers_returns_zero():
    model = TieredCommission(tiers=[])
    assert model.apply(1000, 100.0, 0.01, 0) == pytest.approx(0.0)


def test_tiered_commission_zero_trade():
    model = TieredCommission(tiers=_TIERS)
    assert model.apply(0, 100.0, 0.01, 0) == pytest.approx(0.0)
