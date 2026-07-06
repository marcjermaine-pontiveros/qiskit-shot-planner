"""H2 molecule VQE workload: hardware-efficient ansatz + Pauli-term observables.

The 2-electron H2 Hamiltonian in the STO-3G basis (Jordan-Wigner, frozen core)
reduces on two qubits to five Pauli terms -- four diagonal (Z-basis) and one
off-diagonal exchange term:

    H = c_II * II + c_IZ * IZ + c_ZI * ZI + c_ZZ * ZZ + c_XX * XX

The four Z-basis terms share one measurement circuit, but ``XX`` does not
commute with them and must be estimated in a rotated (X) basis (Hadamard on
both qubits before measurement). This is the genuine multi-basis case the
Bonferroni multi-Pauli guarantee is built for; ``measured_ansatz`` supplies the
per-term basis-change circuit.
"""

from __future__ import annotations

from typing import Callable

from qiskit import QuantumCircuit

# Keystone sampler contract (see backends/samplers.py): bitstring -> ±eigenvalue.
OutcomeMap = Callable[[str], float]

PAULI_LABELS = ("II", "IZ", "ZI", "ZZ", "XX")

# H2 Hamiltonian coefficients per bond length (STO-3G, JW, frozen core).
# The equilibrium (0.74 A) row is the canonical 2-qubit H2 of O'Malley et al.
# (2016), including the XX exchange term; the ground state of this electronic
# Hamiltonian is -1.857 Ha (physical total, adding the 0.72 Ha nuclear
# repulsion at 0.74 A, is the familiar -1.137 Ha). The other bond lengths keep
# the diagonal reduction only (the thesis VQE result is reported at 0.74 A).
# bond_length_angstrom -> {Pauli label: coefficient}
H2_COEFFS: dict[float, dict[str, float]] = {
    0.50: {"II": -0.8126, "IZ": 0.1712, "ZI": -0.2227, "ZZ": -0.2227},
    0.70: {"II": -1.0175, "IZ": 0.3436, "ZI": -0.3745, "ZZ": -0.0523},
    0.74: {"II": -1.0523732, "IZ": 0.3979374, "ZI": -0.3979374,
           "ZZ": -0.0112801, "XX": 0.1809312},  # equilibrium, canonical 2-qubit H2
    1.00: {"II": -1.1361, "IZ": 0.5003, "ZI": -0.4738, "ZZ": 0.0991},
    1.50: {"II": -1.1785, "IZ": 0.6163, "ZI": -0.5432, "ZZ": 0.2507},
}

# Ground-state ansatz angles (theta0, theta1) for the Ry-Ry-CNOT ansatz. The
# 0.74 A angles reach the exact ground state of the five-term Hamiltonian
# (E = -1.857 Ha); the others minimize the diagonal reduction.
H2_ANGLES: dict[float, tuple[float, float]] = {
    0.50: (0.42, 3.14),
    0.70: (0.68, 3.14),
    0.74: (3.3699, 3.1337),
    1.00: (1.05, 3.14),
    1.50: (1.48, 3.14),
}


def vqe_ansatz(t0: float, t1: float) -> QuantumCircuit:
    """Hardware-efficient 2-qubit ansatz: Ry(t0) ⊗ Ry(t1) → CNOT.

    Args:
        t0: Ry rotation angle on qubit 0.
        t1: Ry rotation angle on qubit 1.

    Returns:
        The unmeasured ansatz circuit.
    """
    qc = QuantumCircuit(2)
    qc.ry(t0, 0)
    qc.ry(t1, 1)
    qc.cx(0, 1)
    return qc


def h2_terms(bond_length: float) -> list[tuple[float, str]]:
    """Return the (coefficient, Pauli label) terms of H2 at ``bond_length``.

    Args:
        bond_length: Bond length in angstroms; must be a key of ``H2_COEFFS``.

    Returns:
        List of ``(coeff, label)`` pairs ordered as ``PAULI_LABELS``.

    Raises:
        KeyError: If ``bond_length`` is not a tabulated value.
    """
    coeffs = H2_COEFFS[bond_length]
    return [(coeffs[label], label) for label in PAULI_LABELS if label in coeffs]


def measured_ansatz(ansatz: QuantumCircuit, label: str) -> QuantumCircuit:
    """Append the basis change that makes ``label`` diagonal, then measurement.

    Diagonal terms (``I``/``Z``) need no rotation; an ``X`` needs a Hadamard and
    a ``Y`` needs ``S^dagger`` then Hadamard on that qubit, so the term is read
    from the computational-basis parity. Non-commuting terms (e.g. ``XX``) thus
    require their own circuit -- the multi-basis case.
    """
    n = len(label)
    qc = ansatz.copy()
    for position, pauli in enumerate(label):
        qubit = n - 1 - position  # big-endian label; qubit 0 is rightmost
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
    return qc  # unmeasured (statevector_sampler needs this; hardware adds its own measure)


def pauli_outcome_map(label: str) -> OutcomeMap:
    """Build an outcome map for a Pauli label, read after any basis change.

    Once :func:`measured_ansatz` has rotated ``X``/``Y`` qubits into the
    computational basis, every non-identity factor contributes to the parity, so
    the eigenvalue is the parity over the qubits carrying any non-``I`` Pauli.

    Args:
        label: Pauli string over ``I``/``X``/``Y``/``Z``, e.g. ``"ZZ"`` or ``"XX"``.

    Returns:
        An :data:`OutcomeMap` from bitstring to ±1.
    """
    n = len(label)

    def outcome(bitstring: str) -> float:
        bits = bitstring.replace(" ", "")
        value = 1.0
        for position, pauli in enumerate(label):
            if pauli != "I":
                qubit = n - 1 - position  # label is big-endian; bits[-1] is qubit 0
                value *= (-1.0) ** int(bits[-1 - qubit])
        return value

    return outcome
