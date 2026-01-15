"""Tests for empirical Bernstein math functions."""

import math
import pytest

from qamp_shotplanner.planners.empirical_bernstein import (
    eb_radius,
    eb_radius_modified,
    geom_checkpoints,
    ebs_delta_schedule,
)


def test_eb_radius_monotone_in_n():
    """For fixed variance, radius should decrease with n."""
    R = 2.0  # Range [-1, 1]
    var = 0.25
    delta = 0.01

    r1 = eb_radius(100, R, var, delta)
    r2 = eb_radius(200, R, var, delta)
    r3 = eb_radius(500, R, var, delta)

    assert r1 > r2 > r3


def test_eb_radius_zero_variance():
    """With zero variance, should reduce to second term only."""
    R = 2.0
    var = 0.0
    delta = 0.01
    n = 100

    radius = eb_radius(n, R, var, delta)
    # First term should be 0, only second term remains
    # Second term: R * 3*ln(3/δ) / n
    expected = 2.0 * 3 * math.log(3.0 / delta) / n
    assert radius == pytest.approx(expected)


def test_eb_radius_invalid_n():
    """Should raise ValueError for n < 1."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        eb_radius(0, 2.0, 0.25, 0.01)


def test_eb_radius_invalid_R():
    """Should raise ValueError for R <= 0."""
    with pytest.raises(ValueError, match="R must be > 0"):
        eb_radius(100, 0.0, 0.25, 0.01)


def test_eb_radius_invalid_delta():
    """Should raise ValueError for delta not in (0, 1)."""
    with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
        eb_radius(100, 2.0, 0.25, 0.0)

    with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
        eb_radius(100, 2.0, 0.25, 1.0)


def test_eb_radius_invalid_variance():
    """Should raise ValueError for negative variance."""
    with pytest.raises(ValueError, match="var_biased must be >= 0"):
        eb_radius(100, 2.0, -0.1, 0.01)


def test_eb_radius_modified_with_alpha():
    """Modified radius should decrease with smaller alpha."""
    n = 100
    R = 2.0
    var = 0.25
    delta_k = 0.001

    r1 = eb_radius_modified(n, R, var, delta_k, alpha=1.0)
    r2 = eb_radius_modified(n, R, var, delta_k, alpha=0.5)

    # Smaller alpha gives smaller x, so smaller radius
    assert r2 < r1


def test_eb_radius_modified_alpha_equals_one():
    """Alpha=1 should give valid bound."""
    n = 100
    R = 2.0
    var = 0.25
    delta_k = 0.01

    radius = eb_radius_modified(n, R, var, delta_k, alpha=1.0)
    assert radius > 0


def test_eb_radius_modified_invalid_delta_k():
    """Should raise ValueError for delta_k outside (0, 3)."""
    with pytest.raises(ValueError, match="delta_k must be in \\(0, 3\\)"):
        eb_radius_modified(100, 2.0, 0.25, 0.0)

    with pytest.raises(ValueError, match="delta_k must be in \\(0, 3\\)"):
        eb_radius_modified(100, 2.0, 0.25, 3.0)


def test_eb_radius_modified_invalid_alpha():
    """Should raise ValueError for alpha <= 0."""
    with pytest.raises(ValueError, match="alpha must be > 0"):
        eb_radius_modified(100, 2.0, 0.25, 0.01, alpha=0.0)


def test_geom_checkpoints_start_at_n_min():
    """First checkpoint should be at or after n_min."""
    checkpoints = geom_checkpoints(beta=1.1, n_min=10, n_max=1000)
    assert checkpoints[0] >= 10


def test_geom_checkpoints_cover_range():
    """Checkpoints should cover from n_min to n_max."""
    checkpoints = geom_checkpoints(beta=1.5, n_min=10, n_max=1000)
    assert checkpoints[0] >= 10
    assert checkpoints[-1] == 1000  # Always ends at n_max
    assert all(checkpoints[i] < checkpoints[i + 1] for i in range(len(checkpoints) - 1))


def test_geom_checkpoints_n_min_equals_n_max():
    """When n_min == n_max, should return single checkpoint."""
    checkpoints = geom_checkpoints(beta=1.1, n_min=100, n_max=100)
    assert checkpoints == [100]


def test_geom_checkpoints_beta_affects_spacing():
    """Larger beta should give fewer checkpoints."""
    checkpoints1 = geom_checkpoints(beta=1.05, n_min=10, n_max=1000)
    checkpoints2 = geom_checkpoints(beta=2.0, n_min=10, n_max=1000)

    assert len(checkpoints1) > len(checkpoints2)


def test_geom_checkpoints_invalid_beta():
    """Should raise ValueError for beta <= 1."""
    with pytest.raises(ValueError, match="beta must be > 1"):
        geom_checkpoints(beta=1.0, n_min=10, n_max=100)


def test_geom_checkpoints_invalid_n_min():
    """Should raise ValueError for n_min < 1."""
    with pytest.raises(ValueError, match="n_min must be >= 1"):
        geom_checkpoints(beta=1.1, n_min=0, n_max=100)


def test_geom_checkpoints_n_max_lt_n_min():
    """Should raise ValueError for n_max < n_min."""
    with pytest.raises(ValueError, match="n_max must be >= n_min"):
        geom_checkpoints(beta=1.1, n_min=100, n_max=10)


def test_delta_schedule_uniform():
    """Uniform schedule should return equal values."""
    deltas = ebs_delta_schedule(delta=0.1, K=5)
    assert len(deltas) == 5
    assert all(d == 0.02 for d in deltas)  # 0.1 / 5 = 0.02


def test_delta_schedule_sums_to_delta():
    """Sum of all deltas should equal original delta."""
    delta = 0.05
    for K in [1, 5, 10, 100]:
        deltas = ebs_delta_schedule(delta, K)
        assert sum(deltas) == pytest.approx(delta)


def test_delta_schedule_k_equals_one():
    """K=1 should return single value equal to delta."""
    deltas = ebs_delta_schedule(delta=0.1, K=1)
    assert deltas == [0.1]


def test_delta_schedule_invalid_delta():
    """Should raise ValueError for delta not in (0, 1)."""
    with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
        ebs_delta_schedule(0.0, 5)

    with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
        ebs_delta_schedule(1.0, 5)


def test_delta_schedule_invalid_K():
    """Should raise ValueError for K < 1."""
    with pytest.raises(ValueError, match="K must be >= 1"):
        ebs_delta_schedule(0.1, 0)
