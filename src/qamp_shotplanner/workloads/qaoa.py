"""QAOA MaxCut workload: 2-qubit p=1 circuit with a ZZ cost observable.

Provides the circuit, the ZZ outcome map, and the exact ⟨ZZ⟩ reference for the
QAOA MaxCut energy-estimation experiment (PCSC exp5). The generic sampler in
``backends/samplers.py`` consumes the circuit and outcome map; this module only
supplies the circuit + outcome map + ideal reference.

Cost Hamiltonian H_C = Z⊗Z (single edge, weight 1); mixer H_B = X⊗I + I⊗X.
Defaults γ=0.783, β=0.438 are near the single-edge MaxCut optimum:
⟨ZZ⟩ ≈ 0.9836, σ² ≈ 0.0325 (low variance → large EBS savings).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

GAMMA_DEFAULT = 0.783
BETA_DEFAULT = 0.438


def qaoa_maxcut_circuit(
    gamma: float = GAMMA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> QuantumCircuit:
    """Build the 2-qubit p=1 MaxCut QAOA circuit (no measurement).

    Phase separation exp(-i γ Z⊗Z) is realized as CX · Rz(2γ) · CX; the mixer
    exp(-i β H_B) as Rx(2β) on each qubit.

    Args:
        gamma: Phase-separation angle.
        beta: Mixer angle.

    Returns:
        A 2-qubit circuit with no classical register or measurements.
    """
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cx(0, 1)
    qc.rz(2 * gamma, 1)
    qc.cx(0, 1)
    qc.rx(2 * beta, 0)
    qc.rx(2 * beta, 1)
    return qc


def zz_outcome_map(bitstring: str) -> float:
    """Map a measured bitstring to the Z⊗Z eigenvalue via parity.

    Even parity → +1, odd parity → -1.

    Args:
        bitstring: Measured computational-basis outcome (e.g. "01").

    Returns:
        +1.0 or -1.0.
    """
    return 1.0 - 2.0 * (bitstring.count("1") % 2)


def ideal_zz(
    gamma: float = GAMMA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> float:
    """Exact ⟨Z⊗Z⟩ for the QAOA state via the statevector.

    Args:
        gamma: Phase-separation angle.
        beta: Mixer angle.

    Returns:
        The ideal expectation value in [-1, 1].
    """
    probs = Statevector.from_instruction(qaoa_maxcut_circuit(gamma, beta)).probabilities()
    return float(sum(p * (1 - 2 * (i.bit_count() % 2)) for i, p in enumerate(probs)))
