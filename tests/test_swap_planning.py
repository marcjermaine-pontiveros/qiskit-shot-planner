"""Tests for SWAP test shot planning."""

from qamp_shotplanner.swaptest.planning import plan_shots_for_swap_fidelity
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner


def test_swap_planning_matches_hoeffding():
    """SWAP planning should match Hoeffding with a=-1, b=+1."""
    epsilon_F = 0.02
    delta = 0.01

    shots_swap = plan_shots_for_swap_fidelity(epsilon_F, delta)

    planner = HoeffdingPlanner(
        epsilon_stat=epsilon_F,
        delta=delta,
        a=-1.0,
        b=+1.0,
    )
    shots_hoeffding = planner.planned_shots()

    assert shots_swap == shots_hoeffding


def test_swap_planning_known_value():
    """Test against known value from the report."""
    shots = plan_shots_for_swap_fidelity(epsilon_F=0.02, delta=0.01)
    assert shots == 26492, f"Expected 26492, got {shots}"


def test_swap_planning_monotonicity():
    """Smaller tolerance should require more shots."""
    shots1 = plan_shots_for_swap_fidelity(epsilon_F=0.01, delta=0.01)
    shots2 = plan_shots_for_swap_fidelity(epsilon_F=0.02, delta=0.01)
    assert shots1 > shots2