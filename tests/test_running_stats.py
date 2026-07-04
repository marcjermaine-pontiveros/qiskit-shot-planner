"""Tests for RunningStats."""

import pytest

from qamp_shotplanner.stats.running_stats import RunningStats


def test_running_stats_empty():
    """Empty stats should have zero count and zero variance."""
    stats = RunningStats()
    assert stats.n == 0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0
    assert stats.variance_biased == 0.0
    assert stats.variance_unbiased == 0.0
    assert stats.std == 0.0


def test_running_stats_single_sample():
    """Single sample should have zero variance."""
    stats = RunningStats()
    stats.update(5.0)
    assert stats.n == 1
    assert stats.mean == 5.0
    assert stats.m2 == 0.0
    assert stats.variance_biased == 0.0
    assert stats.std == 0.0


def test_running_stats_variance_known():
    """Test against known variance values."""
    stats = RunningStats()
    # [1, 2, 3, 4, 5] has mean 3
    # Deviations: [-2, -1, 0, 1, 2], squared: [4, 1, 0, 1, 4]
    # m2 = 10, biased variance = 10/5 = 2.0, unbiased = 10/4 = 2.5
    samples = [1, 2, 3, 4, 5]
    for x in samples:
        stats.update(x)

    assert stats.n == 5
    assert stats.mean == 3.0
    assert stats.variance_biased == 10.0 / 5.0  # m2 / n
    assert stats.variance_unbiased == 10.0 / 4.0  # m2 / (n-1)


def test_running_stats_from_samples():
    """Creating from samples should match manual updates."""
    samples = [1, 2, 3, 4, 5]
    stats1 = RunningStats.from_samples(samples)

    stats2 = RunningStats()
    for x in samples:
        stats2.update(x)

    assert stats1.n == stats2.n
    assert stats1.mean == stats2.mean
    assert stats1.m2 == stats2.m2
    assert stats1.variance_biased == stats2.variance_biased


def test_running_stats_merge():
    """Merging two stats should combine correctly."""
    # First set: [1, 2, 3] -> mean=2, m2=2
    stats1 = RunningStats.from_samples([1, 2, 3])

    # Second set: [4, 5, 6] -> mean=5, m2=2
    stats2 = RunningStats.from_samples([4, 5, 6])

    # Combined: [1, 2, 3, 4, 5, 6] -> mean=3.5, m2=17.5
    combined = stats1.merge(stats2)

    assert combined.n == 6
    assert combined.mean == 3.5
    assert combined.m2 == 17.5


def test_running_stats_merge_empty():
    """Merging with empty stats should return the non-empty one."""
    stats1 = RunningStats.from_samples([1, 2, 3])
    stats2 = RunningStats()

    merged = stats1.merge(stats2)
    assert merged.n == stats1.n
    assert merged.mean == stats1.mean
    assert merged.m2 == stats1.m2


def test_running_stats_std():
    """Standard deviation should be sqrt of variance."""
    stats = RunningStats.from_samples([1, 2, 3, 4, 5])
    # variance_biased = 10/5 = 2.0, so std = sqrt(2.0)
    expected_std = (10.0 / 5.0) ** 0.5
    assert stats.std == pytest.approx(expected_std)


def test_running_stats_unbiased_variance_for_single_sample():
    """Unbiased variance should be 0 for single sample (n-1=0 handled)."""
    stats = RunningStats()
    stats.update(5.0)
    assert stats.variance_unbiased == 0.0  # Special case for n < 2


def test_running_stats_from_samples_empty():
    """Creating from empty sequence should give empty stats."""
    stats = RunningStats.from_samples([])
    assert stats.n == 0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0


def test_running_stats_welford_numerical_stability():
    """Welford should handle values with different scales."""
    # Values far from zero
    stats = RunningStats.from_samples([1e6, 1e6 + 1, 1e6 + 2])
    assert stats.mean == 1e6 + 1
    assert stats.variance_biased == pytest.approx(2.0 / 3.0)


def test_from_binary_counts_matches_from_samples():
    """from_binary_counts should match from_samples for equivalent data."""
    count_0, count_1 = 73, 27
    samples = [1.0] * count_0 + [-1.0] * count_1

    stats_samples = RunningStats.from_samples(samples)
    stats_counts = RunningStats.from_binary_counts(count_0, count_1)

    assert stats_samples.n == stats_counts.n
    assert abs(stats_samples.mean - stats_counts.mean) < 1e-10
    assert abs(stats_samples.m2 - stats_counts.m2) < 1e-10


def test_from_binary_counts_empty():
    """Both counts zero should return empty stats."""
    stats = RunningStats.from_binary_counts(0, 0)
    assert stats.n == 0
    assert stats.mean == 0.0
    assert stats.m2 == 0.0


def test_from_binary_counts_single_positive():
    """Only positive count should work."""
    stats = RunningStats.from_binary_counts(10, 0)
    assert stats.n == 10
    assert stats.mean == 1.0
    assert stats.m2 == 0.0  # All same value, no variance


def test_from_binary_counts_single_negative():
    """Only negative count should work."""
    stats = RunningStats.from_binary_counts(0, 10)
    assert stats.n == 10
    assert stats.mean == -1.0
    assert stats.m2 == 0.0


def test_from_binary_counts_custom_values():
    """Custom values should work correctly."""
    stats = RunningStats.from_binary_counts(
        count_positive=3,
        count_negative=1,
        value_positive=0.0,
        value_negative=1.0,
    )
    assert stats.n == 4
    assert abs(stats.mean - 0.25) < 1e-10  # (0+0+0+1)/4 = 0.25


def test_from_binary_counts_negative_raises():
    """Negative counts should raise ValueError."""
    with pytest.raises(ValueError, match="Counts must be non-negative"):
        RunningStats.from_binary_counts(-1, 5)
