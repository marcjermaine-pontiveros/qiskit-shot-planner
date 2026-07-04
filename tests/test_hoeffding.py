"""Tests for Hoeffding planner."""

import pytest
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner


def test_hoeffding_formula():
    """Test Hoeffding shot formula against known values."""
    planner = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
    shots = planner.planned_shots()
    assert shots == 26492


def test_hoeffding_monotonicity_epsilon():
    """Smaller epsilon should require more shots."""
    planner1 = HoeffdingPlanner(epsilon_stat=0.01, delta=0.01, a=-1.0, b=1.0)
    planner2 = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
    assert planner1.planned_shots() > planner2.planned_shots()


def test_hoeffding_monotonicity_delta():
    """Smaller delta should require more shots."""
    planner1 = HoeffdingPlanner(epsilon_stat=0.02, delta=0.001, a=-1.0, b=1.0)
    planner2 = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
    assert planner1.planned_shots() > planner2.planned_shots()


def test_hoeffding_invalid_epsilon():
    """Should raise ValueError for epsilon <= 0."""
    planner = HoeffdingPlanner(epsilon_stat=0.0, delta=0.5, a=-1.0, b=1.0)
    with pytest.raises(ValueError, match="epsilon_stat must be > 0"):
        planner.planned_shots()


def test_hoeffding_invalid_delta_low():
    """Should raise ValueError for delta <= 0."""
    planner = HoeffdingPlanner(epsilon_stat=0.02, delta=0.0, a=-1.0, b=1.0)
    with pytest.raises(ValueError, match="delta must be in \\(0,1\\)"):
        planner.planned_shots()


def test_hoeffding_invalid_delta_high():
    """Should raise ValueError for delta >= 1."""
    planner = HoeffdingPlanner(epsilon_stat=0.02, delta=1.0, a=-1.0, b=1.0)
    with pytest.raises(ValueError, match="delta must be in \\(0,1\\)"):
        planner.planned_shots()


def test_hoeffding_range_dependence():
    """Larger range [a,b] should require more shots."""
    planner1 = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
    planner2 = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-2.0, b=2.0)
    assert planner2.planned_shots() > planner1.planned_shots()