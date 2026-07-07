"""Grouped estimation on a mixed commuting / non-commuting Hamiltonian.

A transverse-field Ising model (TFIM) on a short chain

    H = J · Σ_i Z_i Z_{i+1}  +  h · Σ_i X_i

is the smallest Hamiltonian that is *not* one commuting family: the ZZ couplings
all commute with each other (all-Z basis) and the X fields all commute with each
other (all-X basis), but a ZZ term and an X term sharing a qubit do NOT commute.
So the m = (n-1) + n Pauli terms collapse into exactly **two** qubit-wise-commuting
(QWC) groups.

This is the general form of the hand-built 3-group Kagome example: here the grouping
front-end (:func:`qubitwise_commuting_groups`) discovers the families automatically,
and :func:`grouped_energy_estimate` estimates the energy at the (eps, delta)
guarantee — the union bound over the two groups, not over the m terms.

Qiskit 2 only. Everything runs on the exact statevector.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector

from qamp_shotplanner import (
    HoeffdingPlanner,
    grouped_energy_estimate,
    qubitwise_commuting_groups,
    statevector_value_sampler,
)

N = 6
J, H_FIELD = 1.0, 0.6
EPS, DELTA = 0.1, 0.05


def _zz_label(i: int, j: int) -> str:
    chars = ["I"] * N
    chars[N - 1 - i] = "Z"
    chars[N - 1 - j] = "Z"
    return "".join(chars)


def _x_label(i: int) -> str:
    chars = ["I"] * N
    chars[N - 1 - i] = "X"
    return "".join(chars)


def tfim_terms() -> list[tuple[float, str]]:
    """(coeff, big-endian label) terms of the TFIM on an open chain."""
    zz = [(J, _zz_label(i, i + 1)) for i in range(N - 1)]
    x = [(H_FIELD, _x_label(i)) for i in range(N)]
    return zz + x


def structured_state():
    """A fixed (non-random-at-runtime) prepared state to estimate on."""
    ansatz = efficient_su2(N, entanglement="linear", reps=1).decompose()
    x = np.random.default_rng(3).uniform(0, 2 * np.pi, ansatz.num_parameters)
    return ansatz.assign_parameters(x)


def main() -> None:
    terms = tfim_terms()
    H = SparsePauliOp([label for _, label in terms], [c for c, _ in terms])
    state = structured_state()
    e_exact = float(Statevector(state).expectation_value(H).real)

    groups = qubitwise_commuting_groups(terms)
    print(f"TFIM chain: N={N} qubits, H = J·ΣZZ + h·ΣX  (J={J}, h={H_FIELD})")
    print(f"exact energy ⟨H⟩ = {e_exact:+.4f}")

    print("\n[GROUPING]")
    print(f"  naive per-term Bonferroni : {len(terms)} terms "
          f"(union bound over {len(terms)} items, ‖c‖₁ = {sum(abs(c) for c,_ in terms):.1f})")
    print(f"  QWC groups                : {len(groups)} groups "
          f"(union bound over {len(groups)} items)")
    for g, group in enumerate(groups):
        print(f"    group {g}: basis {group.basis}  "
              f"({len(group.terms)} terms, R_g = {group.range:.1f})")

    def factory(circuit, value_map, g):
        return statevector_value_sampler(circuit, value_map, seed=2000 + g)

    result = grouped_energy_estimate(
        terms, state, EPS, DELTA, factory, split="uniform"
    )

    # A per-term Hoeffding wall for context (the cost grouping avoids).
    m = len(terms)
    l1 = sum(abs(c) for c, _ in terms)
    eps_j = EPS / (2 * l1)
    per_term_cap = m * HoeffdingPlanner(
        epsilon_stat=eps_j, delta=DELTA / m, a=-1.0, b=1.0
    ).planned_shots()

    print("\n[ESTIMATE]")
    for g, (group, r, eps_g) in enumerate(
        zip(result.groups, result.per_group, result.eps_alloc)
    ):
        print(f"  group {g} ({group.basis}): EBS τ = {r.n:,} shots, "
              f"eps_g = {eps_g:.4f}, stopped_by={r.stopped_by}")
    print(f"\n  per-term Bonferroni Hoeffding wall : {per_term_cap:,} shots")
    print(f"  grouped EBS total                  : {result.total_shots:,} shots")
    print(f"  E_hat = {result.energy:+.4f}   "
          f"|E_hat - E_exact| = {abs(result.energy - e_exact):.4f}")
    print(f"  certified bound Σ eps_g = {result.certified_eps:.4f}  (target eps = {EPS})")
    assert abs(result.energy - e_exact) <= EPS, "estimate outside eps (rare, reseed)"
    print("  OK: estimate within eps of the exact energy.")


if __name__ == "__main__":
    main()
