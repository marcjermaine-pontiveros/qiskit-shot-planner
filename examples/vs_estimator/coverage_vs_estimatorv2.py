"""EstimatorV2 default precision has no (eps, delta) guarantee; adaptive stopping does.

The Qiskit ``EstimatorV2`` workflow the utility-scale notebooks use fixes the shot
budget through ``default_precision`` -- a target *standard error* of 1/sqrt(n),
so n = ceil(1/precision^2) shots, chosen WITHOUT reference to the observable's
variance. That is a heuristic, not a coverage guarantee: whether the returned
estimate actually lands within a tolerance eps depends on the observable, and the
primitive never certifies it.

This example makes the gap empirical. For several single-qubit observables
<Z> = cos(theta) (spanning low to high variance sigma^2 = 1 - mu^2), it compares:

  * EstimatorV2 policy: n = ceil(1/precision^2) shots at the default precision
    (0.0156 -> ~4096 shots), report the empirical coverage Pr(|Zhat - mu| <= eps);
  * adaptive EBS: stop at the (eps, delta) Maurer-Pontil radius, report coverage
    and the mean stopping time.

The point: EstimatorV2's coverage swings with variance (it over-covers easy
observables and UNDER-covers hard ones -- no uniform guarantee), while the
adaptive rule holds at or above 1 - delta everywhere, at a variance-adaptive cost.

Qiskit 2 only. Runs on the exact statevector.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    statevector_sampler,
)

EPS, DELTA = 0.02, 0.01
DEFAULT_PRECISION = 0.0156                  # Qiskit EstimatorV2 default
N_EST = math.ceil(1.0 / DEFAULT_PRECISION**2)  # ~4096 shots, variance-agnostic
TRIALS = 400


def _z_outcome(bitstring: str) -> float:
    return 1.0 if bitstring.replace(" ", "")[-1] == "0" else -1.0


def _ry_circuit(mu: float) -> QuantumCircuit:
    """One qubit with <Z> = cos(theta) = mu."""
    qc = QuantumCircuit(1)
    qc.ry(math.acos(mu), 0)
    return qc


def main() -> None:
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    print(f"target (eps, delta) = ({EPS}, {DELTA}); Hoeffding cap n_H = {n_h:,}")
    print(f"EstimatorV2 default precision {DEFAULT_PRECISION} -> fixed n = {N_EST:,} shots\n")
    print(f"{'mu':>5} {'sigma^2':>8} | {'EstV2 cover':>11} | {'adaptive cover':>14} {'mean tau':>9}")
    print("-" * 58)

    for mu in (0.2, 0.5, 0.8, 0.95):
        sigma2 = 1.0 - mu * mu
        qc = _ry_circuit(mu)

        est_hits, ada_hits, taus = 0, 0, []
        for t in range(TRIALS):
            # EstimatorV2 policy: fixed N_EST shots, sample mean, no guarantee
            est_mean = float(np.mean(statevector_sampler(qc, _z_outcome, seed=10_000 + t)(N_EST)))
            est_hits += abs(est_mean - mu) <= EPS

            # Adaptive: stop at the (eps, delta) radius
            r = EmpiricalBernsteinStopper(
                epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
            ).run_batched(statevector_sampler(qc, _z_outcome, seed=20_000 + t))
            ada_hits += abs(r.estimate - mu) <= EPS
            taus.append(r.n)

        print(f"{mu:>5.2f} {sigma2:>8.3f} | {100*est_hits/TRIALS:>10.1f}% | "
              f"{100*ada_hits/TRIALS:>13.1f}% {int(np.mean(taus)):>9,}")

    print("\nEstimatorV2's fixed budget over-covers easy observables and under-covers")
    print(f"hard ones (target is {100*(1-DELTA):.0f}%); adaptive holds >= {100*(1-DELTA):.0f}% throughout.")


if __name__ == "__main__":
    main()
