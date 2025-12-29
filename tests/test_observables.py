"""Tests for observable construction helpers."""

import pytest
from qiskit.quantum_info import SparsePauliOp

from qamp_shotplanner.observables.basics import (
    pauli_x,
    pauli_y,
    pauli_z,
    single_qubit_observable,
)
from qamp_shotplanner.observables.multi_qubit import (
    correlation_observable,
    bell_state_observable,
    hamiltonian_term,
)


class TestSingleQubitObservables:
    """Test single-qubit observable construction."""

    def test_pauli_x_single_qubit(self):
        """Test X observable on single-qubit system."""
        obs = pauli_x(qubit=0, num_qubits=1)
        assert str(obs.paulis[0]) == "X"
        assert obs.coeffs[0] == 1.0

    def test_pauli_y_single_qubit(self):
        """Test Y observable on single-qubit system."""
        obs = pauli_y(qubit=0, num_qubits=1)
        assert str(obs.paulis[0]) == "Y"
        assert obs.coeffs[0] == 1.0

    def test_pauli_z_single_qubit(self):
        """Test Z observable on single-qubit system."""
        obs = pauli_z(qubit=0, num_qubits=1)
        assert str(obs.paulis[0]) == "Z"
        assert obs.coeffs[0] == 1.0

    def test_pauli_x_multi_qubit(self):
        """Test X observable on qubit 0 in 2-qubit system."""
        obs = pauli_x(qubit=0, num_qubits=2)
        # Qiskit endianness: rightmost is qubit 0, so "IX"
        assert str(obs.paulis[0]) == "IX"
        assert obs.coeffs[0] == 1.0

    def test_pauli_z_qubit_1(self):
        """Test Z observable on qubit 1 in 2-qubit system."""
        obs = pauli_z(qubit=1, num_qubits=2)
        # Qiskit endianness: rightmost is qubit 0, so qubit 1 is "ZI"
        assert str(obs.paulis[0]) == "ZI"
        assert obs.coeffs[0] == 1.0

    def test_single_qubit_observable_x(self):
        """Test generic single-qubit observable with X."""
        obs = single_qubit_observable(qubit=0, num_qubits=2, pauli="X")
        assert str(obs.paulis[0]) == "IX"
        assert obs.coeffs[0] == 1.0

    def test_single_qubit_observable_y(self):
        """Test generic single-qubit observable with Y."""
        obs = single_qubit_observable(qubit=0, num_qubits=2, pauli="Y")
        assert str(obs.paulis[0]) == "IY"
        assert obs.coeffs[0] == 1.0

    def test_single_qubit_observable_invalid(self):
        """Test that invalid Pauli raises ValueError."""
        with pytest.raises(ValueError, match="pauli must be 'X', 'Y', or 'Z'"):
            single_qubit_observable(qubit=0, num_qubits=2, pauli="W")


class TestMultiQubitObservables:
    """Test multi-qubit observable construction."""

    def test_correlation_zz(self):
        """Test ZZ correlation observable."""
        obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="Z", pauli2="Z")
        assert str(obs.paulis[0]) == "ZZ"
        assert obs.coeffs[0] == 1.0

    def test_correlation_xx(self):
        """Test XX correlation observable."""
        obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="X", pauli2="X")
        assert str(obs.paulis[0]) == "XX"
        assert obs.coeffs[0] == 1.0

    def test_correlation_yy(self):
        """Test YY correlation observable."""
        obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="Y", pauli2="Y")
        assert str(obs.paulis[0]) == "YY"
        assert obs.coeffs[0] == 1.0

    def test_correlation_mixed(self):
        """Test mixed correlation observable like XZ."""
        obs = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="X", pauli2="Z")
        assert str(obs.paulis[0]) == "ZX"
        assert obs.coeffs[0] == 1.0

    def test_bell_state_zz(self):
        """Test Bell state ZZ observable helper."""
        obs = bell_state_observable(num_qubits=2, correlation_type="ZZ")
        assert str(obs.paulis[0]) == "ZZ"
        assert obs.coeffs[0] == 1.0

    def test_bell_state_xx(self):
        """Test Bell state XX observable helper."""
        obs = bell_state_observable(num_qubits=2, correlation_type="XX")
        assert str(obs.paulis[0]) == "XX"
        assert obs.coeffs[0] == 1.0

    def test_bell_state_yy(self):
        """Test Bell state YY observable helper."""
        obs = bell_state_observable(num_qubits=2, correlation_type="YY")
        assert str(obs.paulis[0]) == "YY"
        assert obs.coeffs[0] == 1.0

    def test_bell_state_invalid(self):
        """Test that invalid correlation type raises ValueError."""
        with pytest.raises(ValueError, match="correlation_type must be"):
            bell_state_observable(num_qubits=2, correlation_type="XY")


class TestHamiltonianTerms:
    """Test Hamiltonian term construction."""

    def test_single_qubit_term(self):
        """Test single-qubit Hamiltonian term."""
        term = hamiltonian_term(qubits=(0,), paulis=("Z",), coefficient=1.0, num_qubits=1)
        assert str(term.paulis[0]) == "Z"
        assert term.coeffs[0] == 1.0

    def test_two_qubit_term(self):
        """Test two-qubit Hamiltonian term."""
        term = hamiltonian_term(qubits=(0, 1), paulis=("Z", "Z"), coefficient=-0.5, num_qubits=2)
        assert str(term.paulis[0]) == "ZZ"
        assert term.coeffs[0] == -0.5

    def test_hamiltonian_term_xx(self):
        """Test XX Hamiltonian term."""
        term = hamiltonian_term(qubits=(0, 1), paulis=("X", "X"), coefficient=0.3, num_qubits=2)
        assert str(term.paulis[0]) == "XX"
        assert term.coeffs[0] == 0.3

    def test_hamiltonian_term_mismatched_lengths(self):
        """Test that mismatched qubits and paulis raises ValueError."""
        with pytest.raises(ValueError, match="qubits and paulis must have same length"):
            hamiltonian_term(qubits=(0, 1), paulis=("Z",), coefficient=1.0, num_qubits=2)

    def test_hamiltonian_term_three_qubits(self):
        """Test three-qubit Hamiltonian term."""
        term = hamiltonian_term(
            qubits=(0, 1, 2), paulis=("X", "Y", "Z"), coefficient=0.1, num_qubits=3
        )
        # Qiskit endianness reverses the order
        assert str(term.paulis[0]) == "ZYX"
        assert term.coeffs[0] == 0.1