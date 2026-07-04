"""Coverage validation for EBS on SWAP test fidelity."""

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator
from qamp_shotplanner.swaptest.run_ebs_estimator import run_swap_fidelity_estimator_ebs


@dataclass
class EBSCoverageStats:
    """Statistics from EBS coverage validation experiments.

    Attributes:
        n_trials: Number of independent trials run
        epsilon_F: Tolerance used for planning
        delta: Target failure probability used for planning
        failures: Number of trials where |F_hat - F_ref| > epsilon_F
        empirical_failure_rate: failures / n_trials
        mean_error: Mean statistical error across all trials
        max_error: Maximum statistical error observed
        mean_shots_used: Average number of shots used by EBS
        max_shots_used: Maximum shots used (cap, i.e., Hoeffding)
    """

    n_trials: int
    epsilon_F: float
    delta: float
    failures: int
    empirical_failure_rate: float
    mean_error: float
    max_error: float
    mean_shots_used: float
    max_shots_used: int


def coverage_validation_swap_ebs(
    theta1: float,
    theta2: float,
    n_trials: int,
    epsilon_F: float,
    delta: float,
    reference_shots: int = 100000,
    noise_model: Optional[NoiseModel] = None,
    beta: float = 1.1,
    alpha: float = 1.0,
) -> EBSCoverageStats:
    """Run coverage validation for SWAP test fidelity estimation with EBS.

    Validates that the EBS-planned shots achieve the promised statistical
    guarantees by running multiple independent trials.

    The EBS guarantee: with probability at least 1 - delta,
    |F_hat - F| ≤ epsilon_F, where F is the true fidelity.

    Args:
        theta1: First state rotation angle
        theta2: Second state rotation angle
        n_trials: Number of independent trials to run
        epsilon_F: Tolerance on fidelity estimate error
        delta: Target failure probability
        reference_shots: Shots for reference value (default large for accuracy)
        noise_model: Optional noise model for simulation
        beta: EBS geometric checkpoint factor (default 1.1)
        alpha: EBS mid-interval tightness parameter (default 1.0)

    Returns:
        EBSCoverageStats with validation results
    """
    qc = swap_test_1qubit(theta1, theta2)

    # Get reference (high-shot) value
    F_ref = run_swap_fidelity_estimator(
        qc,
        shots=reference_shots,
        noise_model=noise_model,
        seed_simulator=9999,
    )

    errors = []
    shots_list = []
    failures = 0

    for trial in range(n_trials):
        F_hat, shots_used = run_swap_fidelity_estimator_ebs(
            qc,
            epsilon_F=epsilon_F,
            delta=delta,
            noise_model=noise_model,
            seed_simulator=trial,
            beta=beta,
            alpha=alpha,
        )

        error = abs(F_hat - F_ref)
        errors.append(error)
        shots_list.append(shots_used)

        if error > epsilon_F:
            failures += 1

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    mean_shots = sum(shots_list) / len(shots_list)
    max_shots = max(shots_list)
    empirical_failure_rate = failures / n_trials

    return EBSCoverageStats(
        n_trials=n_trials,
        epsilon_F=epsilon_F,
        delta=delta,
        failures=failures,
        empirical_failure_rate=empirical_failure_rate,
        mean_error=mean_error,
        max_error=max_error,
        mean_shots_used=mean_shots,
        max_shots_used=max_shots,
    )
