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
