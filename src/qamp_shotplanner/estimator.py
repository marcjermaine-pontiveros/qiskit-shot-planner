"""Drop-in adaptive estimator with a finite-sample (eps, delta) certificate.

Qiskit's ``EstimatorV2`` returns an expectation value at a fixed target *precision*
(a standard error, `1/sqrt(n)`), with no coverage guarantee. ``AdaptiveEstimator``
exposes the same ``run([(circuit, observable), ...])`` shape but instead stops
*adaptively* on the empirical Bernstein radius and returns, per observable, a value
that satisfies ``|value - <O>_device| <= eps`` with probability at least ``1 - delta``,
together with the number of shots that certificate actually cost.

Under the hood it decomposes the observable into Pauli terms, estimates each by the
geometric-checkpoint stopper, and combines them by the Bonferroni union bound
(the identity term is exact and never sampled). The measurement backend enters
only through ``sampler_factory``; the default is the exact statevector sampler, so
the estimator runs out-of-the-box in simulation and accepts a hardware sampler
factory unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qamp_shotplanner.backends.samplers import SampleMany, statevector_sampler
from qamp_shotplanner.planners.bonferroni import bonferroni_estimate
from qamp_shotplanner.workloads.vqe_h2 import measured_ansatz, pauli_outcome_map


@dataclass
class AdaptiveResult:
    """One observable's certified estimate.

    Attributes:
        value: Estimate of ``<observable>``; ``|value - <O>_device| <= eps`` w.p. ``>= 1-delta``.
        shots: Total shots the adaptive rule used to reach that certificate.
        epsilon: Tolerance the estimate is certified to.
        delta: Failure probability of the certificate.
        n_terms: Number of non-identity Pauli terms sampled.
    """

    value: float
    shots: int
    epsilon: float
    delta: float
    n_terms: int = 0


class AdaptiveEstimatorResult:
    """List-like result, indexed per input observable (mirrors EstimatorV2)."""

    def __init__(self, results: list[AdaptiveResult]):
        self._results = results

    def __getitem__(self, i: int) -> AdaptiveResult:
        return self._results[i]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)


class AdaptiveEstimator:
    """EstimatorV2-shaped estimator that certifies ``(eps, delta)`` instead of precision.

    Args:
        epsilon: Target tolerance on the combined observable estimate. Default 0.02.
        delta: Allowed failure probability. Default 0.01.
        sampler_factory: ``(measured_circuit, outcome_map, seed) -> SampleMany``. The
            measured circuit is the state-prep circuit with the term's basis rotation
            appended (no measurement gate — the default statevector sampler needs it
            unmeasured; a hardware factory adds its own). Defaults to the exact
            statevector sampler.
        tight: If True use the tight per-term split ``eps_j = eps/||c||_1`` (exact
            target at a quarter of the shots); else the conservative ``eps/(2||c||_1)``.
    """

    def __init__(
        self,
        epsilon: float = 0.02,
        delta: float = 0.01,
        *,
        sampler_factory=None,
        tight: bool = False,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta
        self.tight = tight
        self._sampler_factory = sampler_factory or (
            lambda circuit, outcome_map, seed: statevector_sampler(
                circuit, outcome_map, seed=seed
            )
        )

    @staticmethod
    def _split(observable: SparsePauliOp) -> tuple[float, list[tuple[float, str]]]:
        """Separate the exact identity constant from the terms that need sampling."""
        const = 0.0
        terms: list[tuple[float, str]] = []
        for label, coeff in observable.to_list():
            c = float(coeff.real)
            if set(label) == {"I"}:
                const += c
            else:
                terms.append((c, label))
        return const, terms

    def run(
        self,
        pubs: list[tuple],
        *,
        seed: int = 0,
        sigmas: list[list[float]] | None = None,
    ) -> AdaptiveEstimatorResult:
        """Estimate each ``(circuit, observable)`` to the certified ``(eps, delta)``.

        Args:
            pubs: List of ``(circuit, observable)`` pairs (extra pub fields are ignored),
                where ``observable`` is a :class:`~qiskit.quantum_info.SparsePauliOp`.
            seed: Base RNG seed; distinct seeds are derived per pub and per term.
            sigmas: Optional per-pub list of per-term standard deviations to enable the
                variance-aware allocation; ``None`` uses the uniform split (no oracle
                variance needed, so it works on any backend).

        Returns:
            An :class:`AdaptiveEstimatorResult`, indexable per input pub.
        """
        results: list[AdaptiveResult] = []
        for idx, pub in enumerate(pubs):
            circuit: QuantumCircuit = pub[0]
            observable: SparsePauliOp = pub[1]
            const, terms = self._split(observable)

            if not terms:  # observable is a pure constant
                results.append(AdaptiveResult(const, 0, self.epsilon, self.delta, 0))
                continue

            samplers: list[tuple[float, SampleMany]] = [
                (
                    coeff,
                    self._sampler_factory(
                        measured_ansatz(circuit, label),
                        pauli_outcome_map(label),
                        seed + 1000 * idx + k,
                    ),
                )
                for k, (coeff, label) in enumerate(terms)
            ]
            res = bonferroni_estimate(
                samplers,
                eps=self.epsilon,
                delta=self.delta,
                tight=self.tight,
                sigmas=(sigmas[idx] if sigmas is not None else None),
            )
            results.append(
                AdaptiveResult(
                    value=res.energy + const,
                    shots=res.total_shots,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    n_terms=len(terms),
                )
            )
        return AdaptiveEstimatorResult(results)
