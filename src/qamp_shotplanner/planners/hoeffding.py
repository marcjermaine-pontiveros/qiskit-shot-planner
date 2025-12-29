"""Hoeffding shot planner for bounded random variables."""

from dataclasses import dataclass
from math import log, ceil


@dataclass
class HoeffdingPlanner:
    """Generic Hoeffding shot planner for a bounded random variable X in [a, b].

    Plans the number of shots required to estimate the mean μ = E[X] within
    epsilon_stat with failure probability delta, using Hoeffding's inequality.

    Attributes:
        epsilon_stat: Tolerance on |hat(μ) - μ| for X itself
        delta: Failure probability (must be in (0, 1))
        a: Lower bound of X
        b: Upper bound of X
    """

    epsilon_stat: float
    delta: float
    a: float
    b: float

    def planned_shots(self) -> int:
        """Calculate the number of shots required by Hoeffding's inequality.

        Returns:
            Number of shots n such that Pr(|hat(μ) - μ| >= epsilon_stat) <= delta

        Raises:
            ValueError: If epsilon_stat <= 0 or delta not in (0, 1)
        """
        if self.epsilon_stat <= 0:
            raise ValueError("epsilon_stat must be > 0")
        if not (0 < self.delta < 1):
            raise ValueError("delta must be in (0,1)")

        R = self.b - self.a
        n = (R**2 / (2 * self.epsilon_stat**2)) * log(2.0 / self.delta)
        return int(ceil(n))