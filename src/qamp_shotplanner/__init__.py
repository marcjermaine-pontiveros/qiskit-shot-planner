"""QAMP Shot Planner - Budgeting circuit runs using concentration bounds."""

__version__ = "0.1.0"

from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner
from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.planning import plan_shots_for_swap_fidelity
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator
from qamp_shotplanner.validation.coverage import coverage_validation_swap, CoverageStats
from qamp_shotplanner.observables import (
    pauli_x,
    pauli_y,
    pauli_z,
    single_qubit_observable,
    correlation_observable,
    bell_state_observable,
    hamiltonian_term,
)

__all__ = [
    # Core planner
    "HoeffdingPlanner",
    # SWAP test
    "swap_test_1qubit",
    "plan_shots_for_swap_fidelity",
    "run_swap_fidelity_estimator",
    # Validation
    "coverage_validation_swap",
    "CoverageStats",
    # Observables
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "single_qubit_observable",
    "correlation_observable",
    "bell_state_observable",
    "hamiltonian_term",
]