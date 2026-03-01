"""Tests for costs/slippage.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data_platform.costs.slippage import (
    LinearSlippage,
    SquareRootImpact,
    VolatilitySlippage,
)

_ALL_MODELS = [
    LinearSlippage(5.0),
    VolatilitySlippage(0.1),
    SquareRootImpact(0.1),
]


# ---------------------------------------------------------------------------
# LinearSlippage
# ---------------------------------------------------------------------------


def test_linear_slippage_known_input():
    """1000 shares × $50 notional × 5 bps = $25."""
    model = LinearSlippage(slippage_bps=5.0)
    cost = model.apply(trade_size=1000, price=50.0, vol=0.01, adv=100_000)
    assert cost == pytest.approx(25.0)


def test_linear_slippage_scales_with_notional():
    model = LinearSlippage(slippage_bps=10.0)
    cost_small = model.apply(100, 10.0, 0.01, 1_000_000)
    cost_large = model.apply(1000, 10.0, 0.01, 1_000_000)
    assert cost_large == pytest.approx(10 * cost_small)


def test_linear_slippage_series_input():
    model = LinearSlippage(slippage_bps=5.0)
    shares = pd.Series([100.0, 200.0, 500.0])
    price = pd.Series([10.0, 10.0, 10.0])
    result = model.apply(shares, price, vol=0.01, adv=1_000_000)
    expected = shares * price * 5.0 / 10_000
    pd.testing.assert_series_equal(pd.Series(result), expected, check_names=False)


# ---------------------------------------------------------------------------
# VolatilitySlippage
# ---------------------------------------------------------------------------


def test_volatility_slippage_scales_with_vol():
    model = VolatilitySlippage(k=0.1)
    low_vol = model.apply(1000, 100.0, vol=0.01, adv=1_000_000)
    high_vol = model.apply(1000, 100.0, vol=0.02, adv=1_000_000)
    assert high_vol == pytest.approx(2 * low_vol)


def test_volatility_slippage_scales_with_participation():
    """Larger trade (higher participation) → higher cost."""
    model = VolatilitySlippage(k=0.1)
    small = model.apply(1_000, 100.0, vol=0.01, adv=1_000_000)
    large = model.apply(4_000, 100.0, vol=0.01, adv=1_000_000)
    # notional is 4×, participation is 4×, sqrt(4)=2 → cost = 4×2 = 8×
    assert large == pytest.approx(8 * small)


def test_volatility_slippage_known_value():
    """participation=0.01, vol=0.01, k=0.1, notional=$100k → cost=$10."""
    model = VolatilitySlippage(k=0.1)
    # 10_000 shares × $100 = $1M notional, adv=1_000_000 → participation=0.01
    cost = model.apply(10_000, 100.0, vol=0.01, adv=1_000_000)
    expected = 1_000_000 * 0.1 * 0.01 * np.sqrt(0.01)
    assert cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SquareRootImpact
# ---------------------------------------------------------------------------


def test_square_root_impact_adv_zero_returns_zero():
    """No crash and zero cost when adv = 0."""
    model = SquareRootImpact(sigma_coeff=0.1)
    cost = model.apply(1000, 50.0, vol=0.02, adv=0)
    assert cost == pytest.approx(0.0)


def test_square_root_impact_known_value():
    """adv_fraction=0.01, vol=0.02, coeff=0.1, notional=$50k → cost=$10."""
    model = SquareRootImpact(sigma_coeff=0.1)
    # 1_000 shares × $50, adv=100_000 → adv_fraction=0.01
    cost = model.apply(1_000, 50.0, vol=0.02, adv=100_000)
    expected = 50_000 * 0.1 * 0.02 * np.sqrt(0.01)
    assert cost == pytest.approx(expected)


def test_square_root_impact_scales_sub_linearly():
    """Doubling trade size → cost increases by sqrt(2), not 2."""
    model = SquareRootImpact(sigma_coeff=0.1)
    base = model.apply(1_000, 100.0, vol=0.01, adv=1_000_000)
    double = model.apply(2_000, 100.0, vol=0.01, adv=1_000_000)
    # notional doubles; adv_fraction doubles → cost = 2 × sqrt(2) × base
    assert double == pytest.approx(2 * np.sqrt(2) * base, rel=1e-6)


# ---------------------------------------------------------------------------
# Common invariants across all models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", _ALL_MODELS)
def test_all_models_non_negative(model):
    cost = model.apply(1000, 50.0, vol=0.01, adv=100_000)
    assert cost >= 0.0


@pytest.mark.parametrize("model", _ALL_MODELS)
def test_all_models_zero_for_zero_trade_size(model):
    cost = model.apply(0, 50.0, vol=0.01, adv=100_000)
    assert cost == pytest.approx(0.0)
