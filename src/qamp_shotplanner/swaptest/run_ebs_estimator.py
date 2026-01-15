"""EBS-based SWAP test fidelity estimation."""

from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import generate_preset_pass_manager

from qamp_shotplanner.planners.ebs_stopping import EmpiricalBernsteinStopper
from qamp_shotplanner.stats.running_stats import RunningStats


def extract_ancilla_counts(counts: dict[str, int]) -> tuple[int, int]:
    """Extract ancilla qubit measurement counts from Qiskit bitstring counts.

    For SWAP test circuits, the ancilla is qubit 0. In Qiskit's big-endian
    bitstring format, qubit 0 is the rightmost character.

    Args:
        counts: Dictionary mapping bitstrings to measurement counts

    Returns:
        (count_0, count_1) where count_0 is the number of ancilla=0 outcomes
        and count_1 is the number of ancilla=1 outcomes
    """
    count_0 = 0
    count_1 = 0
    for bitstring, count in counts.items():
        if bitstring[-1] == "0":
            count_0 += count
        else:
            count_1 += count
    return count_0, count_1


def _run_swap_batch(
    qc: QuantumCircuit,
    shots: int,
    noise_model: Optional[NoiseModel] = None,
    seed_simulator: Optional[int] = None,
) -> tuple[int, int]:
    """Run SWAP test, return counts for ancilla bit measurement.

    Uses AerSimulator.run() to get raw bitstring counts, then extracts
    the ancilla measurement outcomes. For the SWAP test:
        - Bit 0 (ancilla) → +1
        - Bit 1 (ancilla) → -1

    Args:
        qc: SWAP test quantum circuit (3 qubits, ancilla at index 0)
        shots: Number of shots to run
        noise_model: Optional Qiskit noise model for simulation
        seed_simulator: Optional random seed for reproducibility

    Returns:
        (count_0, count_1) where count_0 is number of |0⟩ ancilla outcomes
        and count_1 is number of |1⟩ ancilla outcomes

    Raises:
        AssertionError: If circuit doesn't have exactly 3 qubits
    """
    assert qc.num_qubits == 3, f"Expected 3-qubit circuit, got {qc.num_qubits}"

    backend = AerSimulator()

    backend_opts = {}
    if noise_model is not None:
        backend_opts["noise_model"] = noise_model
    if seed_simulator is not None:
        backend_opts["seed_simulator"] = seed_simulator

    # Add measurements to the circuit (make a copy to avoid modifying original)
    qc_measured = qc.copy()
    qc_measured.measure_all()

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_isa = pm.run(qc_measured)

    # Run with shots to get counts
    result = backend.run(
        qc_isa,
        shots=shots,
        **backend_opts,
    ).result()

    counts = result.get_counts()

    return extract_ancilla_counts(counts)


def run_swap_fidelity_estimator_ebs(
    qc: QuantumCircuit,
    *,
    epsilon_F: float,
    delta: float,
    noise_model: Optional[NoiseModel] = None,
    seed_simulator: Optional[int] = None,
    beta: float = 1.1,
    alpha: float = 1.0,
) -> tuple[float, int]:
    """Run SWAP test with Empirical Bernstein stopping.

    Estimates overlap² F = E[Z_ancilla] where Z_anc ∈ {+1, -1}:
        - Bit 0 → +1
        - Bit 1 → -1

    The EBS algorithm adaptively increases shots until the empirical
    Bernstein bound is below epsilon_F, or until the Hoeffding cap is hit.

    Args:
        qc: SWAP test quantum circuit (3 qubits, ancilla at index 0)
        epsilon_F: Tolerance on fidelity estimate error |F_hat - F|
        delta: Total failure probability
        noise_model: Optional Qiskit noise model for simulation
        seed_simulator: Optional random seed for reproducibility
        beta: Geometric checkpoint factor for EBS (default 1.1)
        alpha: Mid-interval tightness parameter for EBS (default 1.0)

    Returns:
        (F_hat, shots_used) where F_hat is the estimated overlap²
        in [0, 1] and shots_used is the number of shots actually taken

    Raises:
        AssertionError: If circuit doesn't have exactly 3 qubits
    """
    # Create EBS stopper for SWAP test (ancilla outcomes in [-1, 1])
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=epsilon_F,
        delta=delta,
        a=-1.0,  # Per-shot Z_anc ∈ [-1, 1]
        b=+1.0,
        beta=beta,
        alpha=alpha,
    )

    # Track checkpoint index for seed progression
    checkpoint_idx = [0]

    def sample_many(n: int) -> list[float]:
        """Sample n times from the SWAP test circuit."""
        # Use different seed for each checkpoint for independence
        current_seed = None
        if seed_simulator is not None:
            current_seed = seed_simulator + checkpoint_idx[0]
            checkpoint_idx[0] += 1

        count_0, count_1 = _run_swap_batch(
            qc,
            shots=n,
            noise_model=noise_model,
            seed_simulator=current_seed,
        )

        # Convert counts to (+1, -1) samples
        # We don't need to expand the list; we can compute sums directly
        # But for the batched API, we return a list
        samples = [1.0] * count_0 + [-1.0] * count_1
        return samples

    result = stopper.run_batched(sample_many)

    # F_hat = mean of Z_anc outcomes (already in [-1, 1])
    F_hat = result.estimate

    # Clamp to [0, 1] for fidelity (numerical errors could push slightly outside)
    F_hat = max(0.0, min(1.0, F_hat))

    return F_hat, result.n


def run_swap_fidelity_estimator_ebs_batch_optimized(
    qc: QuantumCircuit,
    *,
    epsilon_F: float,
    delta: float,
    noise_model: Optional[NoiseModel] = None,
    seed_simulator: Optional[int] = None,
    beta: float = 1.1,
    alpha: float = 1.0,
) -> tuple[float, int, RunningStats]:
    """Run SWAP test with EBS using optimized batch sampling.

    This is an optimized version that avoids expanding counts into lists.
    Instead, it directly updates RunningStats from the count summaries.

    Args:
        qc: SWAP test quantum circuit (3 qubits, ancilla at index 0)
        epsilon_F: Tolerance on fidelity estimate error |F_hat - F|
        delta: Total failure probability
        noise_model: Optional Qiskit noise model for simulation
        seed_simulator: Optional random seed for reproducibility
        beta: Geometric checkpoint factor for EBS (default 1.1)
        alpha: Mid-interval tightness parameter for EBS (default 1.0)

    Returns:
        (F_hat, shots_used, stats) where F_hat is the estimated overlap²,
        shots_used is the number of shots, and stats contains the full
        RunningStats

    Raises:
        AssertionError: If circuit doesn't have exactly 3 qubits
    """
    assert qc.num_qubits == 3, f"Expected 3-qubit circuit, got {qc.num_qubits}"

    # Create EBS stopper for SWAP test
    stopper = EmpiricalBernsteinStopper(
        epsilon_stat=epsilon_F,
        delta=delta,
        a=-1.0,
        b=+1.0,
        beta=beta,
        alpha=alpha,
    )

    checkpoints = stopper.checkpoints()

    backend = AerSimulator()
    backend_opts = {}
    if noise_model is not None:
        backend_opts["noise_model"] = noise_model

    # Add measurements to the circuit (make a copy to avoid modifying original)
    qc_measured = qc.copy()
    qc_measured.measure_all()

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_isa = pm.run(qc_measured)

    stats = RunningStats()
    prev_checkpoint = 0

    for k, checkpoint in enumerate(checkpoints):
        # Sample up to this checkpoint
        delta_n = checkpoint - prev_checkpoint

        if delta_n > 0:
            # Set seed for this batch
            current_seed = None
            if seed_simulator is not None:
                current_seed = seed_simulator + k

            if current_seed is not None:
                backend_opts["seed_simulator"] = current_seed

            # Run circuit
            result = backend.run(qc_isa, shots=delta_n, **backend_opts).result()
            counts = result.get_counts()

            count_0, count_1 = extract_ancilla_counts(counts)

            # Create batch stats in O(1) space using correct formula
            batch_stats = RunningStats.from_binary_counts(
                count_positive=count_0,
                count_negative=count_1,
                value_positive=1.0,
                value_negative=-1.0,
            )
            # Merge using the mathematically correct parallel Welford algorithm
            stats = stats.merge(batch_stats)
            prev_checkpoint = checkpoint

        # Check stopping criterion using public API
        if stopper.should_stop(stats, checkpoint_index=k):
            F_hat = max(0.0, min(1.0, stats.mean))
            return F_hat, stats.n, stats

    # Hit cap
    F_hat = max(0.0, min(1.0, stats.mean))
    return F_hat, stats.n, stats
