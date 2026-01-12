"""Empirical Bernstein stopping for adaptive shot allocation."""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from qamp_shotplanner.planners.empirical_bernstein import (
    eb_radius_modified,
    ebs_delta_schedule,
    geom_checkpoints,
)
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner
from qamp_shotplanner.stats.running_stats import RunningStats


@dataclass
class StopResult:
    """Result from Empirical Bernstein stopping.

    Attributes:
        estimate: The estimated mean (Ē_N)
        n: Number of samples actually used
        epsilon_n: The empirical Bernstein radius at stopping time
        stats: RunningStats containing mean, variance, etc.
        stopped_by: Either "ebs" (stopped early) or "cap" (hit Hoeffding cap)
    """

    estimate: float
    n: int
    epsilon_n: float
    stats: RunningStats
    stopped_by: Literal["ebs", "cap"]


class EmpiricalBernsteinStopper:
    """Empirical Bernstein stopping for bounded IID variables.

    Uses Algorithm 2 from Gresch–Tepe–Kliesch (arXiv:2502.01730v1):
    - Geometric checking to avoid wasteful evaluations
    - Mid-interval stopping modification (via alpha parameter)
    - Hoeffding cap (never exceeds Hoeffding's planned shots)

    The stopping criterion at checkpoint k:
        ε_n_k < ε_stat

    where ε_n_k is computed using the modified empirical Bernstein bound
    with per-check confidence allocation d_k.
    """

    def __init__(
        self,
        epsilon_stat: float,
        delta: float,
        a: float,
        b: float,
        beta: float = 1.1,
        alpha: float = 1.0,
        n_min: int = 10,
    ):
        """Initialize the EBS stopper.

        Args:
            epsilon_stat: Tolerance on statistical error |hat(μ) - μ|
            delta: Total failure probability
            a: Lower bound of the random variable
            b: Upper bound of the random variable
            beta: Geometric checkpoint factor (> 1). Default 1.1.
            alpha: Mid-interval tightness parameter (> 0). Default 1.0.
            n_min: Minimum samples before first check. Default 10.

        Raises:
            ValueError: If parameters are invalid
        """
        if epsilon_stat <= 0:
            raise ValueError("epsilon_stat must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if a >= b:
            raise ValueError("a must be < b")
        if beta <= 1:
            raise ValueError("beta must be > 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if n_min < 1:
            raise ValueError("n_min must be >= 1")

        self.epsilon_stat = epsilon_stat
        self.delta = delta
        self.a = a
        self.b = b
        self.R = b - a
        self.beta = beta
        self.alpha = alpha
        self.n_min = n_min

        # Compute Hoeffding cap (maximum shots we might use)
        self._hoeffding_planner = HoeffdingPlanner(
            epsilon_stat=epsilon_stat,
            delta=delta,
            a=a,
            b=b,
        )
        self._n_max = self._hoeffding_planner.planned_shots()

        # Generate geometric checkpoints
        self._checkpoints = geom_checkpoints(beta, n_min, self._n_max)
        self._K = len(self._checkpoints)

        # Per-check delta allocation (uniform over checks)
        self._deltas = ebs_delta_schedule(delta, self._K)

    def planned_max_shots(self) -> int:
        """Return the Hoeffding cap (maximum shots that might be used)."""
        return self._n_max

    def checkpoints(self) -> list[int]:
        """Return the geometric checkpoints."""
        return self._checkpoints.copy()

    def run(
        self,
        sample_one: Callable[[], float],
    ) -> StopResult:
        """Run EBS with a single-sample callback.

        This is a convenience wrapper around run_batched for cases where
        samples can only be obtained one at a time.

        Args:
            sample_one: Callable that returns a single sample

        Returns:
            StopResult with the estimate and metadata
        """
        def sample_many(n: int) -> list[float]:
            return [sample_one() for _ in range(n)]

        return self.run_batched(sample_many)

    def run_batched(
        self,
        sample_many: Callable[[int], Sequence[float]],
    ) -> StopResult:
        """Run EBS with batched sampling (preferred for Qiskit).

        Samples are obtained in batches corresponding to geometric checkpoints.
        At each checkpoint, the empirical Bernstein radius is computed and
        compared against epsilon_stat.

        Args:
            sample_many: Callable that takes n and returns n samples

        Returns:
            StopResult with the estimate and metadata
        """
        stats = RunningStats()
        prev_checkpoint = 0

        for k, checkpoint in enumerate(self._checkpoints):
            # Sample up to this checkpoint
            delta_n = checkpoint - prev_checkpoint
            if delta_n > 0:
                new_samples = sample_many(delta_n)
                for x in new_samples:
                    stats.update(x)
                prev_checkpoint = checkpoint

            # Check stopping criterion (only after we have enough samples)
            if stats.n >= self.n_min:
                # Compute modified EB radius at this checkpoint
                delta_k = self._deltas[k]
                epsilon_n = eb_radius_modified(
                    n=stats.n,
                    R=self.R,
                    var_biased=stats.variance_biased,
                    delta_k=delta_k,
                    alpha=self.alpha,
                )

                # Stop if radius is below target
                if epsilon_n < self.epsilon_stat:
                    return StopResult(
                        estimate=stats.mean,
                        n=stats.n,
                        epsilon_n=epsilon_n,
                        stats=stats,
                        stopped_by="ebs",
                    )

        # If we never stopped early, we've hit the Hoeffding cap
        # Compute final radius for reporting
        final_epsilon = eb_radius_modified(
            n=stats.n,
            R=self.R,
            var_biased=stats.variance_biased,
            delta_k=self._deltas[-1],
            alpha=self.alpha,
        )

        return StopResult(
            estimate=stats.mean,
            n=stats.n,
            epsilon_n=final_epsilon,
            stats=stats,
            stopped_by="cap",
        )
