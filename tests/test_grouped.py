"""Tests for qubit-wise-commuting grouping and grouped energy estimation."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qamp_shotplanner import (
    grouped_energy_estimate,
    qubitwise_commuting_groups,
    statevector_value_sampler,
)


def test_tfim_groups_into_two_families():
    """ZZ couplings and X fields form exactly two QWC groups (all-Z, all-X)."""
    # 4-qubit TFIM: ZZ on the chain (all-Z) + X on each qubit (all-X).
    terms = [
        (1.0, "IIZZ"), (1.0, "IZZI"), (1.0, "ZZII"),  # ZZ couplings
        (0.5, "IIIX"), (0.5, "IIXI"), (0.5, "IXII"), (0.5, "XIII"),  # X fields
    ]
    groups = qubitwise_commuting_groups(terms)

    assert len(groups) == 2
    bases = {g.basis for g in groups}
    assert bases == {"ZZZZ", "XXXX"}

    by_basis = {g.basis: g for g in groups}
    assert len(by_basis["ZZZZ"].terms) == 3
    assert len(by_basis["XXXX"].terms) == 4
    # every term is placed exactly once
    assert sum(len(g.terms) for g in groups) == len(terms)


def test_all_commuting_terms_form_one_group():
    """A pure-ZZ Hamiltonian is a single commuting family."""
    terms = [(1.0, "IIZZ"), (1.0, "IZZI"), (1.0, "ZZII"), (2.0, "ZIIZ")]
    groups = qubitwise_commuting_groups(terms)
    assert len(groups) == 1
    assert groups[0].basis == "ZZZZ"


def test_qwc_identity_slots_are_compatible():
    """Terms touching disjoint qubits with different Paulis still share one basis."""
    # X on qubit 0 and Z on qubit 2 act on disjoint qubits -> QWC, basis "ZIX".
    terms = [(1.0, "IIX"), (1.0, "ZII")]
    groups = qubitwise_commuting_groups(terms)
    assert len(groups) == 1
    assert groups[0].basis == "ZIX"


def test_grouped_estimate_within_eps_on_bell_state():
    """Grouped energy matches the exact expectation within eps on a small case."""
    # 2-qubit TFIM-style mix: ZZ (commuting with itself) + X fields.
    terms = [(1.0, "ZZ"), (0.5, "IX"), (0.5, "XI")]
    state = QuantumCircuit(2)
    state.ry(0.7, 0)
    state.cx(0, 1)
    state.ry(0.4, 1)

    H = SparsePauliOp([label for _, label in terms], [c for c, _ in terms])
    e_exact = float(Statevector(state).expectation_value(H).real)

    eps, delta = 0.05, 0.05

    def factory(circuit, value_map, g):
        return statevector_value_sampler(circuit, value_map, seed=100 + g)

    result = grouped_energy_estimate(terms, state, eps, delta, factory)

    assert result.certified_eps <= eps + 1e-12
    assert abs(result.energy - e_exact) <= eps


def test_tight_split_preserves_certified_bound():
    """The range-aware 'tight' split still certifies Σ eps_g = eps."""
    terms = [(1.0, "ZZ"), (0.5, "IX"), (0.5, "XI")]
    state = QuantumCircuit(2)
    state.h(0)
    state.cx(0, 1)

    def factory(circuit, value_map, g):
        return statevector_value_sampler(circuit, value_map, seed=7 + g)

    result = grouped_energy_estimate(terms, state, 0.1, 0.05, factory, split="tight")
    assert abs(result.certified_eps - 0.1) < 1e-9
    assert len(result.groups) == 2
