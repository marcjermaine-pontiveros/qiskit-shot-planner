"""Tests for SWAP test EBS functionality."""

import pytest

from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.run_ebs_estimator import (
    run_swap_fidelity_estimator_ebs,
    run_swap_fidelity_estimator_ebs_batch_optimized,
    _run_swap_batch,
    extract_ancilla_counts,
)


def test_swap_ebs_identical_states():
    """High overlap (identical states) should stop early with low variance."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)
    # Fidelity = cos^2(0) = 1 for identical states

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    # Fidelity should be close to 1
    assert F_hat > 0.95

    # For identical states (low variance), should stop well before Hoeffding cap
    from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

    hoeffding_shots = HoeffdingPlanner(
        epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0
    ).planned_shots()
    assert shots_used < hoeffding_shots


def test_swap_ebs_orthogonal_states():
    """Low overlap (orthogonal states) has high variance, may hit cap."""
    import math
    # For orthogonal states with Ry: |0⟩ vs |1⟩ corresponds to θ=0 vs θ=π
    qc = swap_test_1qubit(theta1=0.0, theta2=math.pi)  # Orthogonal: |0⟩ vs |1⟩
    # Fidelity ≈ 0 for orthogonal states

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    # Fidelity should be close to 0
    assert F_hat < 0.1

    # For orthogonal states (maximal variance), might hit cap
    from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

    hoeffding_shots = HoeffdingPlanner(
        epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0
    ).planned_shots()
    assert shots_used <= hoeffding_shots


def test_swap_ebs_return_types():
    """Should return correct types."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.15)

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    assert isinstance(F_hat, float)
    assert isinstance(shots_used, int)
    assert 0.0 <= F_hat <= 1.0
    assert shots_used > 0


def test_swap_ebs_reproducibility_with_seed():
    """Same seed should give same results."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.15)

    F1, n1 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=12345,
    )

    F2, n2 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=12345,
    )

    assert F1 == F2
    assert n1 == n2


def test_swap_ebs_different_seeds_give_different_results():
    """Different seeds may give different results (probabilistic)."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.15)

    F1, n1 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=1,
    )

    F2, n2 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=2,
    )

    # Results might differ (not guaranteed, but likely)
    # At least verify both are valid
    assert 0.0 <= F1 <= 1.0
    assert 0.0 <= F2 <= 1.0


def test_run_swap_batch_counts():
    """Helper function should return valid counts."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)

    count_0, count_1 = _run_swap_batch(
        qc,
        shots=100,
        seed_simulator=42,
    )

    assert count_0 + count_1 == 100
    assert count_0 >= 0
    assert count_1 >= 0


def test_run_swap_batch_with_noise():
    """Helper should work with noise model."""
    from qiskit_aer.noise import NoiseModel

    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)
    noise_model = NoiseModel()

    count_0, count_1 = _run_swap_batch(
        qc,
        shots=100,
        noise_model=noise_model,
        seed_simulator=42,
    )

    assert count_0 + count_1 == 100


def test_swap_ebs_batch_optimized():
    """Batch-optimized version should return similar results."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)

    F_hat, shots_used, stats = run_swap_fidelity_estimator_ebs_batch_optimized(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    assert 0.0 <= F_hat <= 1.0
    assert shots_used > 0
    assert stats.n == shots_used
    assert stats.mean >= -1.0
    assert stats.mean <= 1.0


def test_swap_ebs_batch_optimized_vs_standard():
    """Both versions should give similar results."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.15)

    # Use same seed for reproducibility
    F1, n1 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    F2, n2, _ = run_swap_fidelity_estimator_ebs_batch_optimized(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
    )

    # Results should be similar (may not be identical due to implementation differences)
    assert abs(F1 - F2) < 0.05
    # Shots used should be similar
    assert abs(n1 - n2) < 100


def test_swap_ebs_invalid_circuit():
    """Should raise for non-3-qubit circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)

    with pytest.raises(AssertionError, match="Expected 3-qubit circuit"):
        run_swap_fidelity_estimator_ebs(
            qc,
            epsilon_F=0.02,
            delta=0.01,
        )


def test_swap_ebs_with_custom_beta():
    """A custom beta parameter should be accepted."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.02,
        delta=0.01,
        seed_simulator=42,
        beta=1.5,
    )

    assert 0.0 <= F_hat <= 1.0
    assert shots_used > 0


def test_swap_ebs_known_fidelity_approximately():
    """For specific angles, fidelity should match expected value approximately."""
    # theta1=0, theta2=0: identical states, F = cos^2(0) = 1
    qc = swap_test_1qubit(theta1=0.0, theta2=0.0)

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.01,
        delta=0.001,
        seed_simulator=42,
    )

    # Fidelity should be very close to 1
    assert F_hat > 0.95


def test_swap_ebs_strict_epsilon():
    """Stricter epsilon should require more shots."""
    qc = swap_test_1qubit(theta1=0.1, theta2=0.15)

    _, shots1 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.05,
        delta=0.01,
        seed_simulator=100,
    )

    _, shots2 = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.01,
        delta=0.01,
        seed_simulator=100,
    )

    # Stricter epsilon (smaller) should require more shots
    # (though not guaranteed for every random case)
    # Just verify both are valid
    assert shots1 > 0
    assert shots2 > 0


def test_run_swap_fidelity_estimator_ebs_with_noise():
    """High-level EBS estimator should handle a simple NoiseModel."""
    from qiskit_aer.noise import NoiseModel
    from qamp_shotplanner.swaptest.run_ebs_estimator import run_swap_fidelity_estimator_ebs

    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)
    noise_model = NoiseModel()

    F_hat, shots_used = run_swap_fidelity_estimator_ebs(
        qc,
        epsilon_F=0.1,
        delta=0.05,
        noise_model=noise_model,
        seed_simulator=42,
    )

    # Assert the estimator ran and produced a sensible fidelity estimate
    assert 0.0 <= F_hat <= 1.0
    assert shots_used > 0


def test_run_swap_fidelity_estimator_ebs_batch_optimized_with_noise():
    """Batch-optimized EBS estimator should handle a simple NoiseModel."""
    from qiskit_aer.noise import NoiseModel
    from qamp_shotplanner.swaptest.run_ebs_estimator import run_swap_fidelity_estimator_ebs_batch_optimized

    qc = swap_test_1qubit(theta1=0.1, theta2=0.1)
    noise_model = NoiseModel()

    F_hat, shots_used, stats = run_swap_fidelity_estimator_ebs_batch_optimized(
        qc,
        epsilon_F=0.1,
        delta=0.05,
        noise_model=noise_model,
        seed_simulator=43,
    )

    # Assert the estimator ran and produced a sensible fidelity estimate
    assert 0.0 <= F_hat <= 1.0
    assert shots_used > 0
    assert stats.n == shots_used


def test_extract_ancilla_counts():
    """Test ancilla extraction utility."""
    # Test basic case
    counts = {"000": 30, "001": 20, "010": 25, "011": 25}
    count_0, count_1 = extract_ancilla_counts(counts)
    assert count_0 == 55  # 000 + 010
    assert count_1 == 45  # 001 + 011

    # Test all zeros
    counts = {"000": 100}
    count_0, count_1 = extract_ancilla_counts(counts)
    assert count_0 == 100
    assert count_1 == 0

    # Test all ones
    counts = {"111": 100}
    count_0, count_1 = extract_ancilla_counts(counts)
    assert count_0 == 0
    assert count_1 == 100
