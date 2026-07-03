"""Bonferroni-split shot allocation for multi-term observables.

Estimates ``E = Σ_j c_j μ_j`` (e.g. a Hamiltonian energy) by running an
Empirical Bernstein stopper independently per term, splitting the global
failure probability and accuracy budget across terms so that the combined
estimate meets ``(eps, delta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from qamp_shotplanner.planners.ebs_stopping import (
    EmpiricalBernsteinStopper,
    StopResult,
)

# Keystone sampler contract (see backends/samplers.py): n -> n i.i.d. ±eigenvalues.
SampleMany = Callable[[int], Sequence[float]]


@dataclass
class BonferroniResult:
    """Result of a Bonferroni-split multi-term estimate.

    Attributes:
        energy: The combined estimate ``Σ_j c_j μ̂_j``.
        total_shots: Total shots used across all terms.
        per_term: Per-term :class:`StopResult`, ordered as the input ``terms``.
    """

    energy: float
    total_shots: int
    per_term: list[StopResult]


def _variance_aware_eps(terms, eps, sigmas):
    """Optimal per-term tolerances that minimize total shots.

    Minimizing the empirical-Bernstein leading cost ``sum_j sigma_j^2 / eps_j^2``
    subject to the joint-error constraint ``sum_j |c_j| eps_j = eps`` gives, by a
    Lagrange argument, ``eps_j proportional to (sigma_j^2 / |c_j|)^{1/3}``. The
    normalization restores ``sum_j |c_j| eps_j = eps``, so the joint (eps, delta)
    guarantee is preserved -- only the split changes, not the union bound. This
    concentrates shots on the high-variance / high-weight terms (e.g. the XX
    exchange term of H2) and skips near-deterministic ones.
    """
    weights = [(abs(c) if c else 0.0) for c, _ in terms]
    raw = [((s * s / w) ** (1.0 / 3.0) if (w > 0 and s > 0) else 0.0)
           for (s, w) in zip(sigmas, weights)]
    denom = sum(w * r for w, r in zip(weights, raw))
    # sigma=0 terms are exact (zero error), so they do not consume the eps budget;
    # give them a nominal eps (they stop at n_min) and normalize over the rest.
    return [eps * r / denom if r > 0 else eps for r in raw]


def bonferroni_estimate(
    terms: list[tuple[float, SampleMany]],
    eps: float,
    delta: float,
    R: float = 2.0,
    tight: bool = False,
    sigmas: Sequence[float] | None = None,
) -> BonferroniResult:
    """Estimate ``Σ_j c_j μ_j`` via per-term Empirical Bernstein stopping.

    The failure budget is split uniformly (``delta_j = delta / m``). The
    accuracy budget uses the thesis ‖c‖₁ split ``eps_j = eps / (s · ‖c‖₁)``
    with ``s = 1`` (``tight``) or ``s = 2`` (default) — this refines the flat
    per-term ``eps`` used in the QCE experiments, so ``Σ_j |c_j| eps_j`` bounds
    the combined error rather than each term hitting ``eps`` on its own.

    Args:
        terms: ``(coeff, sampler)`` pairs, one per observable term.
        eps: Target accuracy on the combined estimate.
        delta: Total failure probability, in ``(0, 1)``.
        R: Range ``b - a`` of the per-shot ±eigenvalues (``2.0`` for [-1, 1]).
        tight: If True use ``eps_j = eps / ‖c‖₁``; else ``eps / (2‖c‖₁)``.

    Returns:
        A :class:`BonferroniResult` with the combined energy, total shots, and
        per-term stopper results.
    """
    m = len(terms)
    delta_j = delta / m
    l1_norm = sum(abs(coeff) for coeff, _ in terms)
    if sigmas is not None:  # variance-aware (optimal) allocation
        eps_js = _variance_aware_eps(terms, eps, sigmas)
    else:                   # uniform ‖c‖₁ split
        eps_flat = eps / ((1.0 if tight else 2.0) * l1_norm)
        eps_js = [eps_flat] * m

    half = R / 2.0
    energy = 0.0
    total_shots = 0
    per_term: list[StopResult] = []

    for (coeff, sampler), eps_j in zip(terms, eps_js):
        stopper = EmpiricalBernsteinStopper(
            epsilon_stat=eps_j,
            delta=delta_j,
            a=-half,
            b=half,
        )
        result = stopper.run_batched(sampler)
        per_term.append(result)
        energy += coeff * result.estimate
        total_shots += result.n

    return BonferroniResult(
        energy=energy,
        total_shots=total_shots,
        per_term=per_term,
    )
