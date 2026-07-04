"""EstimatorV2 execution wrapper for SWAP test fidelity estimation."""

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.transpiler import generate_preset_pass_manager
from typing import Optional


def _z_on_qubit(qubit: int, num_qubits: int) -> SparsePauliOp:
    """Construct Pauli Z observable on a specific qubit.

    Qiskit Pauli labels use big-endian ordering: rightmost character is qubit 0.
    This helper constructs the correct string programmatically.

    Args:
        qubit: Target qubit index (0-based)
        num_qubits: Total number of qubits

    Returns:
        SparsePauliOp with Z on target qubit, I on others
    """
    label = ["I"] * num_qubits
    label[num_qubits - 1 - qubit] = "Z"  # Qiskit endianness: rightmost is qubit 0
    return SparsePauliOp("".join(label))


def run_swap_fidelity_estimator(
    qc: QuantumCircuit,
    *,
    shots: int,
    noise_model: Optional = None,
    seed_simulator: Optional[int] = None,
) -> float:
    """Run SWAP test circuit using Aer EstimatorV2 to estimate overlap-squared.

    Computes overlap² = F = E[Z_ancilla] by constructing a Z observable
    on the ancilla qubit (index 0) and running the AerEstimator primitive.

    For the SWAP test, the ancilla Z expectation equals the overlap-squared:
        F = E[Z_ancilla] = |⟨ψ|φ⟩|² ∈ [0, 1]

    Note on ranges:
    - Per-shot outcomes: Z_anc ∈ {+1, -1}, bounded in [-1, 1] (used for Hoeffding)
    - Expected value: F = E[Z_anc] = overlap² ∈ [0, 1] (what we return)

    Args:
        qc: SWAP test quantum circuit (3 qubits, ancilla at index 0)
        shots: Number of shots to run
        noise_model: Optional Qiskit noise model for simulation
        seed_simulator: Optional random seed for reproducibility

    Returns:
        Estimated overlap² F_hat = E[Z_ancilla] in [0, 1]

    Raises:
        AssertionError: If circuit doesn't have exactly 3 qubits
    """
    assert qc.num_qubits == 3, f"Expected 3-qubit circuit, got {qc.num_qubits}"

    # Construct Z observable on ancilla qubit 0
    # Qiskit endianness: rightmost character is qubit 0, so for qubit 0 in 3-qubit system: "IIZ"
    observable = _z_on_qubit(qubit=0, num_qubits=qc.num_qubits)

    backend = AerSimulator()

    backend_opts = {}
    if noise_model is not None:
        backend_opts["noise_model"] = noise_model
    if seed_simulator is not None:
        backend_opts["seed_simulator"] = seed_simulator

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_isa = pm.run(qc)

    options = {
        "backend_options": backend_opts,
        "run_options": {"shots": shots},
    }

    estimator = AerEstimator(options=options)
    pub = (qc_isa, observable, [])
    job = estimator.run([pub])
    result = job.result()[0]

    return float(result.data.evs)