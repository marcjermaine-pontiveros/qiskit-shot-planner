"""Running statistics using Welford's online algorithm."""

from dataclasses import dataclass
from math import sqrt
from typing import Sequence


@dataclass
class RunningStats:
    """Incremental statistics using Welford's algorithm.

    Tracks count, mean, and sum of squared differences (m2) for online
    computation of mean and variance without storing all samples.

    The variance_biased property returns m2/n (the second moment form used
    in the empirical Bernstein paper).

    Attributes:
        n: Number of samples observed
        mean: Current sample mean
        m2: Sum of squared differences from the current mean
    """

    n: int
    mean: float
    m2: float

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0):
        """Initialize running stats.

        Args:
            n: Initial count (default 0)
            mean: Initial mean (default 0.0)
            m2: Initial sum of squared differences (default 0.0)
        """
        self.n = n
        self.mean = mean
        self.m2 = m2

    def update(self, x: float) -> None:
        """Update stats with a new sample using Welford's algorithm.

        Args:
            x: New sample value
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Merge another RunningStats into this one.

        Uses the parallel algorithm for combining Welford statistics.

        Args:
            other: Another RunningStats to merge

        Returns:
            A new RunningStats combining both sets of samples
        """
        if other.n == 0:
            return RunningStats(self.n, self.mean, self.m2)
        if self.n == 0:
            return RunningStats(other.n, other.mean, other.m2)

        combined_n = self.n + other.n
        combined_mean = (self.n * self.mean + other.n * other.mean) / combined_n
        delta = other.mean - self.mean
        combined_m2 = (
            self.m2
            + other.m2
            + delta**2 * self.n * other.n / combined_n
        )
        return RunningStats(combined_n, combined_mean, combined_m2)

    @property
    def variance_biased(self) -> float:
        """Biased variance (m2 / n).

        This matches the form (1/N) * sum((x - mean)^2) used in the
        empirical Bernstein paper (Eq. 4).

        Returns:
            Biased sample variance, or 0 if n < 1
        """
        if self.n < 1:
            return 0.0
        return self.m2 / self.n

    @property
    def variance_unbiased(self) -> float:
        """Unbiased variance (m2 / (n - 1)).

        Returns:
            Unbiased sample variance, or 0 if n < 2
        """
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        """Standard deviation (sqrt of biased variance).

        Returns:
            Standard deviation, or 0 if n < 1
        """
        return sqrt(self.variance_biased)

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "RunningStats":
        """Create RunningStats from a sequence of samples.

        Args:
            samples: Sequence of sample values

        Returns:
            RunningStats with all samples incorporated
        """
        stats = cls()
        for x in samples:
            stats.update(x)
        return stats

    @classmethod
    def from_binary_counts(
        cls,
        count_positive: int,
        count_negative: int,
        value_positive: float = 1.0,
        value_negative: float = -1.0,
    ) -> "RunningStats":
        """Create RunningStats from counts of two possible outcomes.

        Computes mean and m2 directly from counts without expanding to a list.
        This is O(1) in space rather than O(n).

        For SWAP test: count_positive = count_0 (ancilla=0, value=+1)
                       count_negative = count_1 (ancilla=1, value=-1)

        Args:
            count_positive: Number of occurrences of value_positive
            count_negative: Number of occurrences of value_negative
            value_positive: Value for positive outcome (default +1.0)
            value_negative: Value for negative outcome (default -1.0)

        Returns:
            RunningStats with computed mean and m2

        Raises:
            ValueError: If counts are negative
        """
        if count_positive < 0 or count_negative < 0:
            raise ValueError("Counts must be non-negative")

        n = count_positive + count_negative
        if n == 0:
            return cls()

        # Compute mean directly: sum / n
        total_sum = count_positive * value_positive + count_negative * value_negative
        mean = total_sum / n

        # Compute m2 = sum((x - mean)^2)
        # For binary values: m2 = count_pos * (val_pos - mean)^2 + count_neg * (val_neg - mean)^2
        m2 = (
            count_positive * (value_positive - mean) ** 2
            + count_negative * (value_negative - mean) ** 2
        )

        return cls(n=n, mean=mean, m2=m2)
