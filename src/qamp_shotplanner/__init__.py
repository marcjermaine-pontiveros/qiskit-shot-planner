"""QAMP Shot Planner - Budgeting circuit runs using concentration bounds."""

__version__ = "0.3.0"

# Core
from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner

# EBS
from qamp_shotplanner.planners.ebs_stopping import (
    EmpiricalBernsteinStopper,
    StopResult,
)
from qamp_shotplanner.planners.empirical_bernstein import (
    eb_radius,
    eb_radius_maurer,
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

# Adaptive extensions (anytime, Bonferroni multi-Pauli)
from qamp_shotplanner.planners.anytime import AnytimeEBSStopper, anytime_delta
from qamp_shotplanner.planners.bonferroni import BonferroniResult, bonferroni_estimate
from qamp_shotplanner.planners.grouped import (
    GroupedResult,
    QWCGroup,
    grouped_energy_estimate,
    qubitwise_commuting_groups,
)

# Backends (sampler adapters, noise models, run provenance)
from qamp_shotplanner.backends import (
    RunRecord,
    backend_sampler,
    calibrated_noise_model,
    depolarizing_noise_model,
    fake_montreal_simulator,
    fetch_job,
    noise_model_sampler,
    offline_replay_sampler,
    run_and_record,
    snapshot_calibration,
    statevector_sampler,
    statevector_value_sampler,
    write_manifest,
)

# Workloads (QAOA MaxCut, VQE H2)
from qamp_shotplanner.workloads import (
    H2_ANGLES,
    H2_COEFFS,
    PAULI_LABELS,
    h2_terms,
    ideal_zz,
    pauli_outcome_map,
    qaoa_maxcut_circuit,
    vqe_ansatz,
    measured_ansatz,
    zz_outcome_map,
)

# Error mitigation and lower bounds
from qamp_shotplanner.mitigation import fold_gates, zne_extrapolate
from qamp_shotplanner.lower_bound import (
    le_cam_additive,
    le_cam_lower_bound,
    le_cam_two_point,
)

__all__ = [
    # Core planner
    "HoeffdingPlanner",
    # EBS core
    "EmpiricalBernsteinStopper",
    "StopResult",
    "RunningStats",
    "eb_radius",
    "eb_radius_maurer",
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
    # Adaptive extensions
    "AnytimeEBSStopper",
    "anytime_delta",
    "BonferroniResult",
    "bonferroni_estimate",
    "GroupedResult",
    "QWCGroup",
    "grouped_energy_estimate",
    "qubitwise_commuting_groups",
    # Backends
    "statevector_sampler",
    "statevector_value_sampler",
    "noise_model_sampler",
    "backend_sampler",
    "offline_replay_sampler",
    "fake_montreal_simulator",
    "depolarizing_noise_model",
    "calibrated_noise_model",
    "RunRecord",
    "run_and_record",
    "fetch_job",
    "snapshot_calibration",
    "write_manifest",
    # Workloads
    "qaoa_maxcut_circuit",
    "zz_outcome_map",
    "ideal_zz",
    "vqe_ansatz",
    "measured_ansatz",
    "h2_terms",
    "pauli_outcome_map",
    "H2_COEFFS",
    "H2_ANGLES",
    "PAULI_LABELS",
    # Mitigation and lower bounds
    "fold_gates",
    "zne_extrapolate",
    "le_cam_lower_bound",
    "le_cam_additive",
    "le_cam_two_point",
]