"""Tests for the Le Cam lower bound."""

from math import log

import pytest

from qamp_shotplanner.lower_bound.le_cam import le_cam_additive, le_cam_lower_bound

# SWAP anchor from the thesis worked example.
SWAP_EPS = 0.02
SWAP_DELTA = 0.01
SWAP_SIGMA2 = 0.1187
HOEFFDING_CAP = 26492  # n_H at the SWAP anchor


def test_variance_regime_dominates():
    """Small ε with moderate σ² puts the σ²/ε² term above the R/ε term."""
    bound = le_cam_lower_bound(eps=0.001, delta=0.01, sigma2=1.0, R=2.0)
    log_inv = log(1.0 / 0.01)
    variance_term = 1.0 * log_inv / 0.001**2
    assert bound == pytest.approx((1.0 / 12.0) * variance_term)


def test_range_regime_dominates():
    """Tiny variance leaves the R·log(1/δ)/ε range term in control."""
    bound = le_cam_lower_bound(eps=0.02, delta=0.01, sigma2=1e-6, R=2.0)
    log_inv = log(1.0 / 0.01)
    range_term = 2.0 * log_inv / 0.02
    assert bound == pytest.approx((1.0 / 12.0) * range_term)


def test_swap_anchor_below_hoeffding_cap():
    """At the SWAP anchor the bound is positive and well below n_H."""
    bound = le_cam_lower_bound(SWAP_EPS, SWAP_DELTA, SWAP_SIGMA2, R=2.0)
    assert 0 < bound < HOEFFDING_CAP
    # Variance regime governs here (σ²/ε² term beats R/ε term).
    log_inv = log(1.0 / SWAP_DELTA)
    variance_term = SWAP_SIGMA2 * log_inv / SWAP_EPS**2
    assert bound == pytest.approx((1.0 / 12.0) * variance_term)


def test_monotonic_in_epsilon():
    """Smaller ε cannot lower the bound."""
    tight = le_cam_lower_bound(0.01, 0.01, 1.0, R=2.0)
    loose = le_cam_lower_bound(0.02, 0.01, 1.0, R=2.0)
    assert tight > loose


def test_monotonic_in_delta():
    """Smaller δ cannot lower the bound."""
    strict = le_cam_lower_bound(0.02, 0.001, 1.0, R=2.0)
    lax = le_cam_lower_bound(0.02, 0.01, 1.0, R=2.0)
    assert strict > lax


def test_monotonic_in_sigma2():
    """Larger σ² cannot lower the bound."""
    high = le_cam_lower_bound(0.02, 0.01, 1.0, R=2.0)
    low = le_cam_lower_bound(0.02, 0.01, 0.5, R=2.0)
    assert high > low


def test_additive_between_half_and_full():
    """The additive form lies between half and the full max form."""
    lb = le_cam_lower_bound(SWAP_EPS, SWAP_DELTA, SWAP_SIGMA2, R=2.0)
    add = le_cam_additive(SWAP_EPS, SWAP_DELTA, SWAP_SIGMA2, R=2.0)
    assert 0.5 * lb <= add <= lb


def test_invalid_eps():
    with pytest.raises(ValueError, match="eps must be > 0"):
        le_cam_lower_bound(0.0, 0.01, 1.0)


def test_invalid_delta_high():
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1/4\]"):
        le_cam_lower_bound(0.02, 0.5, 1.0)


def test_invalid_delta_low():
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1/4\]"):
        le_cam_lower_bound(0.02, 0.0, 1.0)


def test_invalid_sigma2():
    with pytest.raises(ValueError, match="sigma2 must be >= 0"):
        le_cam_lower_bound(0.02, 0.01, -1.0)


def test_invalid_R():
    with pytest.raises(ValueError, match="R must be > 0"):
        le_cam_lower_bound(0.02, 0.01, 1.0, R=0.0)
