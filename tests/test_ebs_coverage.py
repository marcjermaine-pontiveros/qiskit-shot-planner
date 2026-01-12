"""Tests for EBS coverage validation."""

from qamp_shotplanner.validation.ebs_coverage import (
    coverage_validation_swap_ebs,
    EBSCoverageStats,
)


def test_ebs_coverage_validation_happy_path():
    """Coverage validation should run successfully with small n_trials."""
    # This is a smoke test to ensure the function works end-to-end
    stats = coverage_validation_swap_ebs(
        theta1=0.1,
        theta2=0.15,
        n_trials=10,  # Small for quick test
        epsilon_F=0.05,
        delta=0.1,  # Larger delta for this quick test
        reference_shots=10000,
    )

    assert isinstance(stats, EBSCoverageStats)
    assert stats.n_trials == 10
    assert stats.epsilon_F == 0.05
    assert stats.delta == 0.1
    assert stats.failures >= 0
    assert stats.failures <= stats.n_trials
    assert 0.0 <= stats.empirical_failure_rate <= 1.0
    assert stats.mean_error >= 0.0
    assert stats.max_error >= 0.0
    assert stats.mean_shots_used > 0
    assert stats.max_shots_used > 0
    assert stats.max_shots_used >= stats.mean_shots_used


def test_ebs_coverage_with_custom_beta_alpha():
    """Coverage validation should accept custom beta and alpha."""
    stats = coverage_validation_swap_ebs(
        theta1=0.1,
        theta2=0.15,
        n_trials=5,
        epsilon_F=0.05,
        delta=0.1,
        reference_shots=5000,
        beta=1.5,
        alpha=0.8,
    )

    assert stats.n_trials == 5
    assert stats.mean_shots_used > 0


def test_ebs_coverage_stats_structure():
    """EBSCoverageStats should have all expected fields."""
    stats = coverage_validation_swap_ebs(
        theta1=0.1,
        theta2=0.15,
        n_trials=5,
        epsilon_F=0.05,
        delta=0.1,
        reference_shots=5000,
    )

    # Check all attributes exist
    assert hasattr(stats, "n_trials")
    assert hasattr(stats, "epsilon_F")
    assert hasattr(stats, "delta")
    assert hasattr(stats, "failures")
    assert hasattr(stats, "empirical_failure_rate")
    assert hasattr(stats, "mean_error")
    assert hasattr(stats, "max_error")
    assert hasattr(stats, "mean_shots_used")
    assert hasattr(stats, "max_shots_used")


def test_ebs_coverage_identical_states_low_variance():
    """For identical states (low variance), mean shots should be relatively low."""
    stats = coverage_validation_swap_ebs(
        theta1=0.1,
        theta2=0.1,  # Identical states
        n_trials=10,
        epsilon_F=0.03,
        delta=0.05,
        reference_shots=10000,
    )

    # For identical states, variance is low so EBS should stop early
    from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

    hoeffding_shots = HoeffdingPlanner(
        epsilon_stat=0.03, delta=0.05, a=-1.0, b=1.0
    ).planned_shots()

    # Mean EBS shots should be less than Hoeffding cap for low variance case
    assert stats.mean_shots_used < hoeffding_shots


def test_ebs_coverage_max_shots_does_not_exceed_hoeffding():
    """max_shots_used should never exceed Hoeffding's planned shots."""
    stats = coverage_validation_swap_ebs(
        theta1=0.1,
        theta2=0.15,
        n_trials=5,
        epsilon_F=0.02,
        delta=0.01,
        reference_shots=5000,
    )

    from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

    hoeffding_shots = HoeffdingPlanner(
        epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0
    ).planned_shots()

    # EBS should never exceed the Hoeffding cap
    assert stats.max_shots_used <= hoeffding_shots
    # And should have used at least the minimum samples
    assert stats.max_shots_used >= 10
