"""SWAP test circuit construction."""

from qiskit import QuantumCircuit


def swap_test_1qubit(theta1: float, theta2: float) -> QuantumCircuit:
    """Construct a 3-qubit SWAP test circuit for single-qubit states.

    Creates a SWAP test circuit that estimates the overlap-squared between two
    single-qubit states prepared by Ry rotations.

    Args:
        theta1: Rotation angle for state |psi> on qubit 1
        theta2: Rotation angle for state |phi> on qubit 2

    Returns:
        QuantumCircuit: 3-qubit circuit where:
            - Qubit 0: Ancilla (measured via Pauli Z observable)
            - Qubit 1: |psi> = Ry(theta1)|0>
            - Qubit 2: |phi> = Ry(theta2)|0>

    Note:
        No measurements are added to the circuit. The Estimator primitive
        will measure the Pauli Z observable on the ancilla qubit.

        The SWAP test estimates overlap-squared: F = E[Z_ancilla] = |⟨ψ|φ⟩|²
        - Per-shot outcomes: Z_anc ∈ {+1, -1} (bounded in [-1, 1])
        - Expected value: F = E[Z_anc] = overlap² ∈ [0, 1]
        - Identical states: F ≈ 1
        - Orthogonal states: F ≈ 0
    """
    qc = QuantumCircuit(3)

    qc.ry(theta1, 1)
    qc.ry(theta2, 2)

    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)

    return qc