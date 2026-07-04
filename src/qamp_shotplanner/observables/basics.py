"""Single-qubit observable construction helpers."""

from qiskit.quantum_info import SparsePauliOp
from typing import Literal


def pauli_x(qubit: int, num_qubits: int) -> SparsePauliOp:
    """Construct Pauli X observable on a specific qubit.

    Args:
        qubit: Target qubit index (0-based)
        num_qubits: Total number of qubits in the system

    Returns:
        SparsePauliOp with X on target qubit, I on others

    Example:
        >>> obs = pauli_x(qubit=0, num_qubits=2)
        >>> # Returns: IX (measures X on qubit 0)
    """
    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit] = "X"
    return SparsePauliOp("".join(label))


def pauli_y(qubit: int, num_qubits: int) -> SparsePauliOp:
    """Construct Pauli Y observable on a specific qubit.

    Args:
        qubit: Target qubit index (0-based)
        num_qubits: Total number of qubits in the system

    Returns:
        SparsePauliOp with Y on target qubit, I on others

    Example:
        >>> obs = pauli_y(qubit=1, num_qubits=2)
        >>> # Returns: YI (measures Y on qubit 1)
    """
    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit] = "Y"
    return SparsePauliOp("".join(label))


def pauli_z(qubit: int, num_qubits: int) -> SparsePauliOp:
    """Construct Pauli Z observable on a specific qubit.

    Args:
        qubit: Target qubit index (0-based)
        num_qubits: Total number of qubits in the system

    Returns:
        SparsePauliOp with Z on target qubit, I on others

    Example:
        >>> obs = pauli_z(qubit=0, num_qubits=2)
        >>> # Returns: IZ (measures Z on qubit 0)

    Note:
        For any single-qubit observable, each measurement yields ±1.
        Therefore, the per-shot outcomes are bounded in [-1, 1].
    """
    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit] = "Z"
    return SparsePauliOp("".join(label))


def single_qubit_observable(
    qubit: int,
    num_qubits: int,
    pauli: Literal["X", "Y", "Z"],
) -> SparsePauliOp:
    """Generic single-qubit Pauli observable constructor.

    Args:
        qubit: Target qubit index (0-based)
        num_qubits: Total number of qubits in the system
        pauli: Pauli operator ("X", "Y", or "Z")

    Returns:
        SparsePauliOp with specified Pauli on target qubit, I on others

    Raises:
        ValueError: If pauli is not "X", "Y", or "Z"

    Example:
        >>> obs = single_qubit_observable(qubit=0, num_qubits=3, pauli="X")
        >>> obs = single_qubit_observable(qubit=1, num_qubits=3, pauli="Y")
        >>> obs = single_qubit_observable(qubit=2, num_qubits=3, pauli="Z")

    Note:
        All Pauli observables have eigenvalues ±1, so per-shot outcomes
        are bounded in [-1, 1]. Hoeffding's inequality applies directly.
    """
    if pauli not in ["X", "Y", "Z"]:
        raise ValueError(f"pauli must be 'X', 'Y', or 'Z', got '{pauli}'")

    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit] = pauli
    return SparsePauliOp("".join(label))