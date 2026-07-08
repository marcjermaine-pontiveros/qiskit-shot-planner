"""Drop-in adaptive sampler with a finite-sample (eps, delta) certificate.

Qiskit's ``SamplerV2`` draws a fixed number of shots and returns the raw outcomes,
with no statistical claim attached. A concentration inequality has nothing to bind
to raw draws -- it certifies a *bounded scalar*. ``AdaptiveSampler`` names that
scalar: the probability that a shot lands in a target outcome set ``S``. Each shot
becomes the indicator ``1[outcome in S] in {0, 1}`` -- a Bernoulli -- so the same
empirical Bernstein stopper used by :class:`AdaptiveEstimator` applies directly, and
the sampler stops as soon as ``|prob - P(S)_device| <= eps`` holds with probability
at least ``1 - delta``, reporting the shots that certificate cost.

This is the honest ``SamplerV2`` counterpart: it certifies a *functional* of the
output distribution (a target-outcome probability), not the full high-dimensional
distribution, whose certification would scale with support size and forfeit the
variance advantage. The measurement backend enters only through ``sampler_factory``;
the default reads the exact statevector, so it runs out-of-the-box in simulation and
accepts a hardware sampler factory unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from qamp_shotplanner.backends.samplers import SampleMany, statevector_value_sampler
from qamp_shotplanner.planners.ebs_stopping import EmpiricalBernsteinStopper


@dataclass
class SamplerResult:
    """One target set's certified probability.

    Attributes:
        probability: Estimate of ``P(outcome in targets)``; certified to within
            ``epsilon`` of the device probability with probability ``>= 1 - delta``.
        shots: Total shots the adaptive rule used to reach that certificate.
        epsilon: Tolerance the probability is certified to.
        delta: Failure probability of the certificate.
        targets: The outcome bitstrings counted as a success.
    """

    probability: float
    shots: int
    epsilon: float
    delta: float
    targets: tuple[str, ...] = ()


class AdaptiveSamplerResult:
    """List-like result, indexed per input pub (mirrors SamplerV2)."""

    def __init__(self, results: list[SamplerResult]):
        self._results = results

    def __getitem__(self, i: int) -> SamplerResult:
        return self._results[i]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)


class AdaptiveSampler:
    """SamplerV2-shaped sampler that certifies a target-outcome probability.

    Where :class:`AdaptiveEstimator` certifies ``<O>``, this certifies
    ``P(outcome in targets)`` to ``(eps, delta)`` -- the same empirical Bernstein
    rule on the Bernoulli indicator of each shot. Low- and high-probability targets
    have small variance, so the rule stops well before a fixed budget; a target near
    ``p = 1/2`` costs the most, exactly as the variance dependence predicts.

    Args:
        epsilon: Target tolerance on the probability. Default 0.02.
        delta: Allowed failure probability. Default 0.01.
        sampler_factory: ``(circuit, value_map, seed) -> SampleMany``, where
            ``value_map`` sends a computational-basis bitstring to its ``{0, 1}``
            success indicator. Defaults to the exact statevector sampler.
        n_min: Minimum shots before the first stopping check. Default 10.

    Note:
        Bitstrings follow Qiskit ordering (qubit 0 is the rightmost character), the
        same convention as ``format(index, "0Nb")`` over the statevector index. Pass
        an unmeasured circuit, or one with final measurements -- they are stripped and
        the state is read in the computational basis.
    """

    def __init__(
        self,
        epsilon: float = 0.02,
        delta: float = 0.01,
        *,
        sampler_factory=None,
        n_min: int = 10,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta
        self.n_min = n_min
        self._sampler_factory = sampler_factory or (
            lambda circuit, value_map, seed: statevector_value_sampler(
                circuit, value_map, seed=seed
            )
        )

    def run(
        self,
        pubs: list[tuple],
        *,
        seed: int = 0,
    ) -> AdaptiveSamplerResult:
        """Certify ``P(outcome in targets)`` for each ``(circuit, targets)`` pub.

        Args:
            pubs: List of ``(circuit, targets)`` pairs, where ``targets`` is a single
                bitstring (e.g. ``"11"``) or an iterable of bitstrings; a shot is a
                success if its outcome equals any of them.
            seed: Base RNG seed; a distinct seed is derived per pub.

        Returns:
            An :class:`AdaptiveSamplerResult`, indexable per input pub.
        """
        results: list[SamplerResult] = []
        for idx, pub in enumerate(pubs):
            circuit: QuantumCircuit = pub[0]
            target = pub[1]
            targets = (
                (target,)
                if isinstance(target, str)
                else tuple(target)  # iterable of bitstrings
            )
            target_set = frozenset(targets)
            prepared = circuit.remove_final_measurements(inplace=False)

            def value_map(bits: str, _t: frozenset = target_set) -> float:
                return 1.0 if bits in _t else 0.0

            sampler: SampleMany = self._sampler_factory(
                prepared, value_map, seed + 1000 * idx
            )
            stopper = EmpiricalBernsteinStopper(
                epsilon_stat=self.epsilon,
                delta=self.delta,
                a=0.0,
                b=1.0,
                n_min=self.n_min,
            )
            res = stopper.run_batched(sampler)
            results.append(
                SamplerResult(
                    probability=res.estimate,
                    shots=res.n,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    targets=targets,
                )
            )
        return AdaptiveSamplerResult(results)
