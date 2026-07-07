"""Kagome Heisenberg energy estimation — where per-term Bonferroni hits its limit.

Motivated by the IBM Open Science Prize 2022 Kagome VQE problem, reframed as a
shot-allocation study. The 12-site Kagome unit cell Heisenberg Hamiltonian

    H = sum_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)          (18 edges -> 54 terms)

is a genuinely many-term observable. Estimating its energy exposes a limit of
the thesis's per-term Bonferroni guarantee, and the fix that lifts it:

1. LIMIT — naive per-term Bonferroni splits the tolerance as eps_j = eps/(2||c||_1)
   with ||c||_1 = 54, so each term must be resolved ~||c||_1^2 times more tightly
   than a single observable. The per-term Hoeffding cap explodes to ~5.4e8 shots;
   the estimate is not simulable and not runnable on hardware.

2. RESOLUTION — the 54 terms fall into 3 commuting families (all-X, all-Y, all-Z),
   each read from ONE basis circuit so the 18 edges in a family share shots. We
   estimate the intensive energy density e = E/N (a bounded [-1.5, 1.5] observable),
   for which the fixed budget is modest and EBS gives a clean reduction.

Qiskit 2 only. Everything runs on the exact statevector.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    statevector_value_sampler,
)

N = 12
# 12-site Kagome unit cell (edge list from the Open Science Prize 2022 notebook)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
         (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8),
         (3, 9), (4, 9), (4, 10), (5, 10), (5, 11), (0, 11)]
BASES = ("X", "Y", "Z")
EPS, DELTA = 0.02, 0.01


def _pauli_label(i: int, j: int, pauli: str) -> str:
    """Big-endian Pauli string with ``pauli`` on qubits i and j, identity elsewhere."""
    chars = ["I"] * N
    chars[N - 1 - i] = pauli
    chars[N - 1 - j] = pauli
    return "".join(chars)


def hamiltonian() -> SparsePauliOp:
    labels = [_pauli_label(i, j, p) for (i, j) in EDGES for p in BASES]
    return SparsePauliOp(labels)


def structured_state():
    """A low-energy prepared circuit from a light statevector optimization.

    This stands in for a converged VQE state; reproducing the optimizer is not
    the point (the library estimates a fixed observable, not the outer loop).
    """
    # decompose once so each Statevector call runs on flat gates (fast)
    ansatz = efficient_su2(N, entanglement="linear", reps=2).decompose()
    H = hamiltonian()

    def energy(x: np.ndarray) -> float:
        return float(Statevector(ansatz.assign_parameters(x)).expectation_value(H).real)

    x0 = np.random.default_rng(7).uniform(0, 2 * np.pi, ansatz.num_parameters)
    x_opt = minimize(energy, x0, method="COBYLA", options={"maxiter": 250}).x
    return ansatz.assign_parameters(x_opt)


def _group_value_map(basis: str):
    """Per-shot intensive value in the given basis: (sum_edges parity) / N."""
    def value(bitstring: str) -> float:
        bits = bitstring.replace(" ", "")
        total = 0
        for i, j in EDGES:
            bi = int(bits[len(bits) - 1 - i])
            bj = int(bits[len(bits) - 1 - j])
            total += (-1) ** (bi ^ bj)
        return total / N
    return value


def _rotated_circuit(prep, basis: str):
    """Copy the state-prep circuit and rotate every qubit into the ``basis``."""
    qc = prep.copy()
    for q in range(N):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
    return qc


def naive_per_term_budget() -> int:
    """The wall: per-term Hoeffding cap under the eps_j = eps/(2||c||_1) split."""
    m = 3 * len(EDGES)                      # 54 terms
    eps_j = EPS / (2 * m)                   # ||c||_1 = m (unit coefficients)
    per_term = HoeffdingPlanner(
        epsilon_stat=eps_j, delta=DELTA / m, a=-1.0, b=1.0
    ).planned_shots()
    return per_term, per_term * m


def main() -> None:
    H = hamiltonian()
    e_ground = float(np.linalg.eigvalsh(H.to_matrix())[0]) / N
    print(f"Kagome Heisenberg: {3 * len(EDGES)} terms on {N} qubits; exact E/N = {e_ground:+.4f}")

    per_term, total = naive_per_term_budget()
    print("\n[LIMIT] naive per-term Bonferroni (eps_j = eps/(2||c||_1), ||c||_1 = 54):")
    print(f"        per-term Hoeffding cap = {per_term:,} shots  ->  {total:,} total")
    print("        ~||c||_1^2 = 2916x the single-observable cost; not simulable/runnable.")

    state = structured_state()
    e_ref = float(Statevector(state).expectation_value(H).real) / N
    print(f"\n[RESOLUTION] 3 commuting families, intensive energy density e = E/N")
    print(f"             structured-state e_ref = {e_ref:+.4f}")

    eps_g = EPS / (2 * len(BASES))          # uniform split over the 3 groups
    n_h = len(BASES) * HoeffdingPlanner(
        epsilon_stat=eps_g, delta=DELTA / len(BASES), a=-1.5, b=1.5
    ).planned_shots()

    total_shots, e_hat = 0, 0.0
    for k, basis in enumerate(BASES):
        sampler = statevector_value_sampler(
            _rotated_circuit(state, basis), _group_value_map(basis), seed=1000 + k
        )
        stopper = EmpiricalBernsteinStopper(
            epsilon_stat=eps_g, delta=DELTA / len(BASES), a=-1.5, b=1.5
        )
        r = stopper.run_batched(sampler)
        total_shots += r.n
        e_hat += r.estimate
        print(f"             group {basis}: EBS tau = {r.n:,}")

    print(f"\n             fixed Hoeffding baseline : {n_h:,} shots")
    print(f"             EBS-geom (grouped)      : {total_shots:,} shots  ({n_h / total_shots:.1f}x)")
    print(f"             e_hat = {e_hat:+.4f}  |e_hat - e_ref| = {abs(e_hat - e_ref):.4f}  (target {EPS})")


if __name__ == "__main__":
    main()
