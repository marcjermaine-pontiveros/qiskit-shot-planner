"""Tests for EmpiricalBernsteinStopper."""

import random

import pytest

from qamp_shotplanner.planners.ebs_stopping import (
    EmpiricalBernsteinStopper,
    StopResult,
)


def test_ebs_never_exceeds_hoeffding_cap():
    """EBS should never use more shots than Hoeffding planner."""
    random.seed(42)
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    # With a high-variance sampler, should hit cap
    def sample_many(n: int):
        # Maximum variance distribution on [-1, 1]: half at -1, half at 1
        return [1.0 if random.random() < 0.5 else -1.0 for _ in range(n)]

    result = stopper.run_batched(sample_many)

    assert result.n <= stopper.planned_max_shots()


def test_ebs_stopper_with_fake_sampler_low_variance():
    """Low variance sampler should stop early."""
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
        beta=1.5,  # Larger beta = fewer checkpoints
    )

    # Constant sampler (zero variance) should stop at first checkpoint
    def sample_many(n: int):
        return [0.5 for _ in range(n)]

    result = stopper.run_batched(sample_many)

    # Should stop well before Hoeffding cap
    assert result.n < stopper.planned_max_shots()
    assert result.stopped_by == "ebs"
    assert result.epsilon_n < stopper.epsilon_stat


def test_ebs_stopper_with_fake_sampler_high_variance():
    """High variance sampler should hit cap."""
    random.seed(123)
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.01,  # Tight epsilon
        delta=0.01,
        a=-1.0,
        b=1.0,
        beta=1.05,  # Small beta = many checkpoints
    )

    # Maximum variance sampler
    def sample_many(n: int):
        return [1.0 if random.random() < 0.5 else -1.0 for _ in range(n)]

    result = stopper.run_batched(sample_many)

    # Should hit Hoeffding cap
    assert result.n == stopper.planned_max_shots()
    assert result.stopped_by == "cap"


def test_ebs_result_structure():
    """StopResult should have all expected fields."""
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    def sample_many(n: int):
        return [0.0 for _ in range(n)]

    result = stopper.run_batched(sample_many)

    assert hasattr(result, "estimate")
    assert hasattr(result, "n")
    assert hasattr(result, "epsilon_n")
    assert hasattr(result, "stats")
    assert hasattr(result, "stopped_by")
    assert result.stopped_by in ["ebs", "cap"]


def test_ebs_checkpoints():
    """Checkpoints should be generated correctly."""
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
        beta=1.5,
        n_min=10,
    )

    checkpoints = stopper.checkpoints()

    assert len(checkpoints) > 0
    assert checkpoints[0] >= 10
    assert checkpoints[-1] == stopper.planned_max_shots()
    assert all(checkpoints[i] < checkpoints[i + 1] for i in range(len(checkpoints) - 1))


def test_ebs_planned_max_shots():
    """planned_max_shots should equal Hoeffding's planned shots."""
    from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

    epsilon = 0.02
    delta = 0.01
    a, b = -1.0, 1.0

    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=epsilon,
        delta=delta,
        a=a,
        b=b,
    )

    hoeffding = HoeffdingPlanner(epsilon_stat=epsilon, delta=delta, a=a, b=b)

    assert stopper.planned_max_shots() == hoeffding.planned_shots()


def test_ebs_invalid_epsilon():
    """Should raise ValueError for epsilon <= 0."""
    with pytest.raises(ValueError, match="epsilon_stat must be > 0"):
        EmpiricalBernsteinStopper(
            epsilon_stat=0.0,
            delta=0.01,
            a=-1.0,
            b=1.0,
        )


def test_ebs_invalid_delta():
    """Should raise ValueError for delta not in (0, 1)."""
    with pytest.raises(ValueError, match="delta must be in \\(0, 1\\)"):
        EmpiricalBernsteinStopper(
            epsilon_stat=0.02,
            delta=0.0,
            a=-1.0,
            b=1.0,
        )


def test_ebs_invalid_bounds():
    """Should raise ValueError for a >= b."""
    with pytest.raises(ValueError, match="a must be < b"):
        EmpiricalBernsteinStopper(
            epsilon_stat=0.02,
            delta=0.01,
            a=1.0,
            b=1.0,
        )


def test_ebs_invalid_beta():
    """Should raise ValueError for beta <= 1."""
    with pytest.raises(ValueError, match="beta must be > 1"):
        EmpiricalBernsteinStopper(
            epsilon_stat=0.02,
            delta=0.01,
            a=-1.0,
            b=1.0,
            beta=1.0,
        )


def test_ebs_invalid_n_min():
    """Should raise ValueError for n_min < 1."""
    with pytest.raises(ValueError, match="n_min must be >= 1"):
        EmpiricalBernsteinStopper(
            epsilon_stat=0.02,
            delta=0.01,
            a=-1.0,
            b=1.0,
            n_min=0,
        )


def test_ebs_run_with_single_sample_callback():
    """run() should work with single-sample callback."""
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    # Use a deterministic sequence - make it long enough
    call_count = [0]

    def sample_one():
        call_count[0] += 1
        return 0.5

    result = stopper.run(sample_one)

    assert result.n >= 1
    assert result.stopped_by == "ebs"
    # Should have been called exactly result.n times
    assert call_count[0] == result.n


def test_ebs_estimate_accuracy():
    """With a known mean, estimate should converge."""
    random.seed(999)
    true_mean = 0.7
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.05,
        delta=0.01,
        a=0.0,
        b=1.0,
    )

    # Low variance sampler
    def sample_many(n: int):
        # Values clustered around true_mean with small variance
        return [true_mean + random.gauss(0, 0.01) for _ in range(n)]

    result = stopper.run_batched(sample_many)

    # Estimate should be close to true mean
    assert abs(result.estimate - true_mean) < stopper.epsilon_stat


def test_delta_schedule_returns_copy():
    """delta_schedule should return a copy for immutability."""
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    deltas1 = stopper.delta_schedule()
    deltas2 = stopper.delta_schedule()

    # Should be equal but not same object
    assert deltas1 == deltas2
    assert deltas1 is not deltas2

    # Modifying one shouldn't affect the other
    deltas1[0] = 999.0
    assert deltas2[0] != 999.0


def test_delta_schedule_sums_to_delta():
    """Sum of delta schedule should equal total delta."""
    delta = 0.05
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=delta,
        a=-1.0,
        b=1.0,
    )

    deltas = stopper.delta_schedule()
    assert abs(sum(deltas) - delta) < 1e-10


def test_compute_radius_matches_manual():
    """compute_radius should match manual calculation."""
    from qamp_shotplanner.planners.empirical_bernstein import eb_radius_maurer
    from qamp_shotplanner.stats.running_stats import RunningStats

    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    stats = RunningStats.from_samples([0.5] * 100)

    # Manual calculation using the provable two-sided Maurer-Pontil radius,
    # which is what compute_radius evaluates.
    deltas = stopper.delta_schedule()
    expected = eb_radius_maurer(
        n=stats.n,
        R=stopper.R,
        var_biased=stats.variance_biased,
        delta=deltas[0],
    )

    # API calculation
    actual = stopper.compute_radius(stats, checkpoint_index=0)

    assert abs(actual - expected) < 1e-10


def test_should_stop_respects_n_min():
    """should_stop should return False before n_min samples."""
    from qamp_shotplanner.stats.running_stats import RunningStats

    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
        n_min=50,
    )

    # Create stats with fewer than n_min samples
    stats = RunningStats.from_samples([0.5] * 10)  # Only 10 samples

    # Should not stop even with low variance
    assert stopper.should_stop(stats, checkpoint_index=0) is False


def test_compute_radius_invalid_index():
    """compute_radius should raise IndexError for invalid checkpoint_index."""
    import pytest
    from qamp_shotplanner.stats.running_stats import RunningStats

    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=0.02,
        delta=0.01,
        a=-1.0,
        b=1.0,
    )

    stats = RunningStats.from_samples([0.5] * 100)

    with pytest.raises(IndexError):
        stopper.compute_radius(stats, checkpoint_index=999)
