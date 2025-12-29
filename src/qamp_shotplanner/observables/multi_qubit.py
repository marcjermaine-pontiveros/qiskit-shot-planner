"""Multi-qubit observable construction helpers."""

from qiskit.quantum_info import SparsePauliOp
from typing import Literal, Tuple


def correlation_observable(
    qubit1: int,
    qubit2: int,
    num_qubits: int,
    pauli1: Literal["X", "Y", "Z"] = "Z",
    pauli2: Literal["X", "Y", "Z"] = "Z",
) -> SparsePauliOp:
    """Construct two-qubit correlation observable.

    Creates an observable for measuring the correlation between two qubits,
    e.g., ZZ correlation, XX correlation, or mixed correlations like XZ.

    Args:
        qubit1: First qubit index (0-based)
        qubit2: Second qubit index (0-based)
        num_qubits: Total number of qubits in the system
        pauli1: Pauli operator for qubit 1 ("X", "Y", or "Z")
        pauli2: Pauli operator for qubit 2 ("X", "Y", or "Z")

    Returns:
        SparsePauliOp representing the correlation observable

    Example:
        >>> obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2)
        >>> # Returns: IZ ⊗ IZ = ZZ (measures ZZ correlation)
        >>> obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="X", pauli2="X")
        >>> # Returns: IX ⊗ IX = XX (measures XX correlation)

    Note:
        Correlation observables have eigenvalues ±1, so per-shot outcomes
        are bounded in [-1, 1]. The correlation ranges from -1 (anti-correlated)
        to +1 (correlated), with 0 indicating no correlation.

        For product states: E[O₁⊗O₂] = E[O₁] · E[O₂]
        For entangled states: Can exhibit correlations impossible classically
    """
    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit1] = pauli1
    label[num_qubits - 1 - qubit2] = pauli2
    return SparsePauliOp("".join(label))


def bell_state_observable(
    num_qubits: int,
    correlation_type: Literal["XX", "YY", "ZZ"] = "ZZ",
) -> SparsePauliOp:
    """Construct correlation observable for Bell state analysis.

    Creates a two-qubit correlation observable suitable for analyzing
    Bell state entanglement. Typically used on qubits 0 and 1.

    Args:
        num_qubits: Total number of qubits in the system
        correlation_type: Type of correlation ("XX", "YY", or "ZZ")

    Returns:
        SparsePauliOp for the specified correlation

    Raises:
        ValueError: If correlation_type is not "XX", "YY", or "ZZ"

    Example:
        >>> obs = bell_state_observable(num_qubits=2, correlation_type="ZZ")
        >>> # Returns: ZZ observable
        >>> obs = bell_state_observable(num_qubits=2, correlation_type="XX")
        >>> # Returns: XX observable

    Note:
        For the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
        - E[ZZ] = +1 (perfect positive correlation)
        - E[XX] = +1 (perfect positive correlation)
        - E[YY] = -1 (perfect negative correlation)

        These correlations violate Bell inequalities and demonstrate
        quantum entanglement.
    """
    if correlation_type not in ["XX", "YY", "ZZ"]:
        raise ValueError(
            f"correlation_type must be 'XX', 'YY', or 'ZZ', got '{correlation_type}'"
        )

    pauli_char = correlation_type[0]  # Extract 'X', 'Y', or 'Z'
    return correlation_observable(0, 1, num_qubits, pauli_char, pauli_char)


def hamiltonian_term(
    qubits: Tuple[int, ...],
    paulis: Tuple[Literal["X", "Y", "Z"], ...],
    coefficient: float = 1.0,
    num_qubits: int = 2,
) -> SparsePauliOp:
    """Construct a single term of a Hamiltonian as a Pauli observable.

    Hamiltonians can be decomposed into sums of Pauli terms:
    H = Σ c_i · P_i where P_i are tensor products of Pauli operators.

    Args:
        qubits: Tuple of qubit indices (0-based)
        paulis: Tuple of Pauli operators ("X", "Y", or "Z") for each qubit
        coefficient: Numerical coefficient for this term
        num_qubits: Total number of qubits in the system

    Returns:
        SparsePauliOp representing the Hamiltonian term

    Raises:
        ValueError: If qubits and paulis have different lengths

    Example:
        >>> # Transverse Ising model: H = Z₀ + 0.5·X₀X₁
        >>> term1 = hamiltonian_term(qubits=(0,), paulis=("Z",), coefficient=1.0, num_qubits=2)
        >>> term2 = hamiltonian_term(qubits=(0, 1), paulis=("X", "X"), coefficient=0.5, num_qubits=2)

    Note:
        Each Hamiltonian term P_i has eigenvalues ±1 (for normalized Paulis),
        so per-shot measurements are bounded in [-|c|, +|c|] where c is the coefficient.

        For shot planning, apply Hoeffding bounds to each term separately:
        - Term with coefficient c: bounded in [-|c|, +|c|]
        - Plan shots for each term independently
        - Combine results with error propagation
    """
    if len(qubits) != len(paulis):
        raise ValueError(
            f"qubits and paulis must have same length, got {len(qubits)} and {len(paulis)}"
        )

    label = ["I"] * num_qubits
    for qubit, pauli in zip(qubits, paulis):
        label[num_qubits - 1 - qubit] = pauli

    return SparsePauliOp("".join(label), coeffs=[coefficient])