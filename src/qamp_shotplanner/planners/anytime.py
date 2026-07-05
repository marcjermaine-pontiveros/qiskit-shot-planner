"""Anytime empirical Bernstein stopping via the π² pointwise schedule.

Harvested from the PCSC 2026 experiments (exp1/exp2 ``run_anytime_ebs``). The
anytime bound splits the total failure probability δ across every sample count n
with the schedule δ_n = 6δ/(π²n²), whose sum over n ≥ 1 equals δ. The empirical
Bernstein radius is therefore valid *simultaneously* at all n, so the criterion
may be checked after every sample rather than only at geometric checkpoints. This
is more conservative at large n than the uniform δ/K schedule and reproduces the
paper's result that anytime-EBS degrades to the Hoeffding cap for
moderate-variance observables.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Sequence

from qamp_shotplanner.planners.ebs_stopping import StopResult
from qamp_shotplanner.planners.empirical_bernstein import eb_radius_maurer
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner
from qamp_shotplanner.stats.running_stats import RunningStats


def anytime_delta(delta: float, n: int) -> float:
    """Per-sample confidence budget for the anytime schedule.

    Args:
        delta: Total failure probability (in (0, 1)).
        n: Sample count (>= 1).

    Returns:
        δ_n = 6δ / (π² n²).
    """
    return 6.0 * delta / (pi**2 * n**2)


class AnytimeEBSStopper:
    """Anytime empirical Bernstein stopping for bounded IID variables.

    Checks the empirical Bernstein radius after every sample past ``n_min``,
    spending the per-sample budget ``anytime_delta(delta, n)``. Never exceeds the
    Hoeffding planned shot count (the cap).
    """

    def __init__(
        self,
        epsilon_stat: float,
        delta: float,
        a: float,
        b: float,
        n_min: int = 10,
    ):
        """Initialize the anytime stopper.

        Args:
            epsilon_stat: Tolerance on the statistical error |hat(μ) - μ|.
            delta: Total failure probability.
            a: Lower bound of the random variable.
            b: Upper bound of the random variable.
            n_min: Minimum samples before the first check. Default 10.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if epsilon_stat <= 0:
            raise ValueError("epsilon_stat must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if a >= b:
            raise ValueError("a must be < b")
        if n_min < 1:
            raise ValueError("n_min must be >= 1")

        self.epsilon_stat = epsilon_stat
        self.delta = delta
        self.a = a
        self.b = b
        self.R = b - a
        self.n_min = n_min

        self._hoeffding = HoeffdingPlanner(
            epsilon_stat=epsilon_stat, delta=delta, a=a, b=b
        )
        self._n_max = self._hoeffding.planned_shots()

    def planned_max_shots(self) -> int:
        """Return the Hoeffding cap (maximum shots that might be used)."""
        return self._n_max

    def _radius(self, stats: RunningStats) -> float:
        return eb_radius_maurer(
            n=stats.n,
            R=self.R,
            var_biased=stats.variance_biased,
            delta=anytime_delta(self.delta, stats.n),
        )

    def run(self, sample_one: Callable[[], float]) -> StopResult:
        """Run with a single-sample callback (wraps ``run_batched``)."""
        return self.run_batched(lambda n: [sample_one() for _ in range(n)])

    def run_batched(
        self,
        sample_many: Callable[[int], Sequence[float]],
    ) -> StopResult:
        """Run anytime EBS with batched sampling.

        Draws ``n_min`` samples, then one sample at a time, checking the radius
        after each until it drops to ``epsilon_stat`` or the Hoeffding cap is hit.

        Args:
            sample_many: Callable taking n and returning n samples.

        Returns:
            StopResult with the estimate and metadata.
        """
        stats = RunningStats()
        for x in sample_many(min(self.n_min, self._n_max)):
            stats.update(x)

        while True:
            if stats.n >= self.n_min:
                epsilon_n = self._radius(stats)
                if epsilon_n <= self.epsilon_stat:
                    return StopResult(
                        estimate=stats.mean,
                        n=stats.n,
                        epsilon_n=epsilon_n,
                        stats=stats,
                        stopped_by="ebs",
                    )
            if stats.n >= self._n_max:
                break
            for x in sample_many(1):
                stats.update(x)

        return StopResult(
            estimate=stats.mean,
            n=stats.n,
            epsilon_n=self._radius(stats),
            stats=stats,
            stopped_by="cap",
        )
