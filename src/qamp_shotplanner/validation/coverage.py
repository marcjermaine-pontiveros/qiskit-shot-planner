"""Coverage validation for Hoeffding bound on SWAP test fidelity."""

from dataclasses import dataclass
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.planning import plan_shots_for_swap_fidelity
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator


@dataclass
class CoverageStats:
    """Statistics from coverage validation experiments.

    Attributes:
        n_trials: Number of independent trials run
        shots_per_trial: Shots used per trial (from Hoeffding planning)
        epsilon_F: Tolerance used for planning
        delta: Target failure probability used for planning
        failures: Number of trials where |F_hat - F_ref| > epsilon_F
        empirical_failure_rate: failures / n_trials
        mean_error: Mean statistical error across all trials
        max_error: Maximum statistical error observed
    """

    n_trials: int
    shots_per_trial: int
    epsilon_F: float
    delta: float
    failures: int
    empirical_failure_rate: float
    mean_error: float
    max_error: float


def coverage_validation_swap(
    theta1: float,
    theta2: float,
    n_trials: int,
    epsilon_F: float,
    delta: float,
    reference_shots: int = 100000,
    noise_model: Optional[NoiseModel] = None,
) -> CoverageStats:
    """Run coverage validation for SWAP test fidelity estimation.

    Validates that the Hoeffding-planned shots achieve the promised statistical
    guarantees by running multiple independent trials.

    Args:
        theta1: First state rotation angle
        theta2: Second state rotation angle
        n_trials: Number of independent trials to run
        epsilon_F: Tolerance on fidelity estimate error
        delta: Target failure probability
        reference_shots: Shots for reference value (default small for speed)
        noise_model: Optional noise model for simulation

    Returns:
        CoverageStats with validation results
    """
    qc = swap_test_1qubit(theta1, theta2)

    shots = plan_shots_for_swap_fidelity(epsilon_F, delta)

    F_ref = run_swap_fidelity_estimator(
        qc,
        shots=reference_shots,
        noise_model=noise_model,
        seed_simulator=9999,
    )

    errors = []
    failures = 0

    for trial in range(n_trials):
        F_hat = run_swap_fidelity_estimator(
            qc,
            shots=shots,
            noise_model=noise_model,
            seed_simulator=trial,
        )

        error = abs(F_hat - F_ref)
        errors.append(error)

        if error > epsilon_F:
            failures += 1

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    empirical_failure_rate = failures / n_trials

    return CoverageStats(
        n_trials=n_trials,
        shots_per_trial=shots,
        epsilon_F=epsilon_F,
        delta=delta,
        failures=failures,
        empirical_failure_rate=empirical_failure_rate,
        mean_error=mean_error,
        max_error=max_error,
    )