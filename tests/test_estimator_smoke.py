"""Smoke test for EstimatorV2 execution wrapper."""

import math
import pytest
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator


def test_estimator_smoke():
    """Basic smoke test: runs without crashing and returns value in [0, 1]."""
    qc = swap_test_1qubit(0.3, 0.8)

    result = run_swap_fidelity_estimator(
        qc,
        shots=2000,
        noise_model=None,
        seed_simulator=42,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0  # overlap² should be in [0, 1]


def test_estimator_with_noise():
    """Test with simple noise model."""
    qc = swap_test_1qubit(0.3, 0.8)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01, 1),
        ["ry", "h"],
    )

    result = run_swap_fidelity_estimator(
        qc,
        shots=5000,
        noise_model=noise_model,
        seed_simulator=42,
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0  # overlap² should be in [0, 1]


def test_estimator_reproducibility():
    """Same seed should give same result."""
    qc = swap_test_1qubit(0.3, 0.8)

    result1 = run_swap_fidelity_estimator(
        qc,
        shots=5000,
        noise_model=None,
        seed_simulator=123,
    )

    result2 = run_swap_fidelity_estimator(
        qc,
        shots=5000,
        noise_model=None,
        seed_simulator=123,
    )

    assert result1 == result2


def test_estimator_circuit_validation():
    """Should raise AssertionError for wrong-sized circuit."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)

    with pytest.raises(AssertionError, match="Expected 3-qubit circuit"):
        run_swap_fidelity_estimator(qc, shots=1000)


def test_estimator_approximate_correctness():
    """Result should be approximately correct using statevector reference."""
    theta1 = 0.3
    theta2 = 0.8
    qc = swap_test_1qubit(theta1, theta2)

    # Get the exact statevector value using the correct observable
    from qiskit.quantum_info import Statevector, Operator
    from qiskit.quantum_info import SparsePauliOp

    sv = Statevector(qc)

    # Use the same helper function to ensure consistency
    from qamp_shotplanner.swaptest.run_estimator import _z_on_qubit
    observable = _z_on_qubit(qubit=0, num_qubits=3)
    observable_matrix = Operator(observable)
    expected = sv.expectation_value(observable_matrix).real

    result = run_swap_fidelity_estimator(
        qc,
        shots=100000,
        noise_model=None,
        seed_simulator=42,
    )

    # Check sampling result is close to exact value
    assert result == pytest.approx(expected, abs=0.02)


def test_swap_identical_states_fidelity_one():
    """Killer test: identical states should give overlap² ≈ 1."""
    theta = 0.7
    qc = swap_test_1qubit(theta, theta)

    result = run_swap_fidelity_estimator(
        qc,
        shots=50000,
        noise_model=None,
        seed_simulator=42,
    )

    # Identical states should have overlap² ≈ 1
    # If this fails, the observable is on the wrong qubit!
    assert result >= 0.95, f"Expected F≈1 for identical states, got {result:.4f}"


def test_swap_orthogonal_states_fidelity_zero():
    """Killer test: orthogonal states should give overlap² ≈ 0."""
    theta1 = 0.0
    theta2 = math.pi  # |0⟩ and |1⟩ are orthogonal
    qc = swap_test_1qubit(theta1, theta2)

    result = run_swap_fidelity_estimator(
        qc,
        shots=50000,
        noise_model=None,
        seed_simulator=42,
    )

    # Orthogonal states should have overlap² ≈ 0
    # If this fails, the observable is on the wrong qubit!
    assert result <= 0.05, f"Expected F≈0 for orthogonal states, got {result:.4f}"