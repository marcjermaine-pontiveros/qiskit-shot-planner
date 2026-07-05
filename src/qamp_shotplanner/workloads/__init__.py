"""End-to-end workloads: SWAP test, QAOA MaxCut, VQE H2."""

from qamp_shotplanner.workloads.qaoa import (
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    ideal_zz,
    qaoa_maxcut_circuit,
    zz_outcome_map,
)
from qamp_shotplanner.workloads.vqe_h2 import (
    H2_ANGLES,
    H2_COEFFS,
    PAULI_LABELS,
    h2_terms,
    pauli_outcome_map,
    vqe_ansatz, measured_ansatz,
)

__all__ = [
    "qaoa_maxcut_circuit",
    "zz_outcome_map",
    "ideal_zz",
    "GAMMA_DEFAULT",
    "BETA_DEFAULT",
    "vqe_ansatz",
    "measured_ansatz",
    "h2_terms",
    "pauli_outcome_map",
    "H2_COEFFS",
    "H2_ANGLES",
    "PAULI_LABELS",
]
