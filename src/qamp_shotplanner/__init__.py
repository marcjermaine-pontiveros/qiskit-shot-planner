"""QAMP Shot Planner - Budgeting circuit runs using concentration bounds."""

__version__ = "0.1.0"

# Core
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

# EBS
from qamp_shotplanner.planners.ebs_stopping import (
    EmpiricalBernsteinStopper,
    StopResult,
)
from qamp_shotplanner.planners.empirical_bernstein import (
    eb_radius,
    eb_radius_modified,
    geom_checkpoints,
    ebs_delta_schedule,
)
from qamp_shotplanner.stats.running_stats import RunningStats

# SWAP test
from qamp_shotplanner.swaptest.circuit import swap_test_1qubit
from qamp_shotplanner.swaptest.planning import plan_shots_for_swap_fidelity
from qamp_shotplanner.swaptest.run_estimator import run_swap_fidelity_estimator
from qamp_shotplanner.swaptest.run_ebs_estimator import (
    run_swap_fidelity_estimator_ebs,
    run_swap_fidelity_estimator_ebs_batch_optimized,
)

# Validation
from qamp_shotplanner.validation.coverage import coverage_validation_swap, CoverageStats
from qamp_shotplanner.validation.ebs_coverage import (
    coverage_validation_swap_ebs,
    EBSCoverageStats,
)

# Observables
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
    # EBS core
    "EmpiricalBernsteinStopper",
    "StopResult",
    "RunningStats",
    "eb_radius",
    "eb_radius_modified",
    "geom_checkpoints",
    "ebs_delta_schedule",
    # SWAP test
    "swap_test_1qubit",
    "plan_shots_for_swap_fidelity",
    "run_swap_fidelity_estimator",
    "run_swap_fidelity_estimator_ebs",
    "run_swap_fidelity_estimator_ebs_batch_optimized",
    # Validation
    "coverage_validation_swap",
    "CoverageStats",
    "coverage_validation_swap_ebs",
    "EBSCoverageStats",
    # Observables
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "single_qubit_observable",
    "correlation_observable",
    "bell_state_observable",
    "hamiltonian_term",
]