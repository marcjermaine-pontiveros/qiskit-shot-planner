"""Sampler factories feeding the adaptive shot planners.

Each factory returns a ``SampleMany`` — a callable that draws ``n`` i.i.d. ±1
outcomes for a fixed observable. The observable enters only through an
``OutcomeMap`` (measured bitstring → ±1 eigenvalue), so the same sampler code
serves SWAP-test fidelity, QAOA ⟨ZZ⟩, VQE Pauli terms, etc.

A ``SampleMany`` plugs directly into
:meth:`~qamp_shotplanner.planners.ebs_stopping.EmpiricalBernsteinStopper.run_batched`
(or ``HoeffdingPlanner``); the planners own all EBS/Hoeffding math.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

SampleMany = Callable[[int], list[float]]
"""Draw ``n`` i.i.d. samples, each a ±eigenvalue in [-1, 1]."""

OutcomeMap = Callable[[str], float]
"""Map a measured bitstring to the observable's ±1 eigenvalue."""

ValueMap = Callable[[str], float]
"""Map a measured bitstring to an arbitrary real (bounded) observable value."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ensure_measured(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy with ``measure_all`` appended if the circuit has no measurements."""
    if any(instr.operation.name == "measure" for instr in circuit.data):
        return circuit
    measured = circuit.copy()
    measured.measure_all()
    return measured


def _counts_to_samples(counts: dict[str, int], outcome_map: OutcomeMap) -> list[float]:
    """Expand a counts dict into a flat list of ±1 eigenvalues via ``outcome_map``."""
    samples: list[float] = []
    for bitstring, count in counts.items():
        value = outcome_map(bitstring.replace(" ", ""))
        samples.extend([value] * count)
    return samples


# ── Statevector (exact, noiseless) — harvest exp5 make_zz_sampler_qaoa ─────────

def statevector_sampler(
    circuit: QuantumCircuit,
    outcome_map: OutcomeMap,
    *,
    seed: int,
) -> SampleMany:
    """Exact-distribution sampler from the noiseless statevector.

    Computes ``P(+1)`` once from ``Statevector`` probabilities, then draws
    numpy Bernoulli samples. This avoids AerSimulator seed-reuse correlations
    across EBS checkpoint batches (the rng state advances between draws).

    Args:
        circuit: Unmeasured circuit (measurements would break ``Statevector``).
        outcome_map: Bitstring → ±1 eigenvalue for the target observable.
        seed: Seed for the numpy Bernoulli generator.

    Returns:
        A ``SampleMany`` drawing ±1 outcomes from the exact distribution.
    """
    n_qubits = circuit.num_qubits
    probs = Statevector.from_instruction(circuit).probabilities()
    p_plus = float(
        sum(p for i, p in enumerate(probs)
            if outcome_map(format(i, f"0{n_qubits}b")) > 0)
    )
    rng = np.random.default_rng(seed)

    def sample_many(n: int) -> list[float]:
        raw = rng.binomial(1, p_plus, size=int(n))
        return (2.0 * raw - 1.0).tolist()

    return sample_many


def statevector_value_sampler(
    circuit: QuantumCircuit,
    value_map: ValueMap,
    *,
    seed: int,
) -> SampleMany:
    """Exact-distribution sampler for a general multi-valued bounded observable.

    Generalizes :func:`statevector_sampler` beyond the two-outcome ±1 case:
    precomputes the observable value for every computational-basis state once,
    then draws basis states from the exact statevector distribution and returns
    their values. Use for commuting-group observables — e.g. the summed parity
    of many Pauli terms measured in one basis — whose per-shot value takes more
    than two levels.

    The precompute is ``O(2**n_qubits)`` in time and memory, so this targets
    small simulated systems (roughly ``n_qubits <= 20``).

    Args:
        circuit: Unmeasured circuit, already rotated into the measurement basis.
        value_map: Bitstring → real observable value for that outcome.
        seed: Seed for the numpy generator.

    Returns:
        A ``SampleMany`` drawing observable values from the exact distribution.
    """
    n_qubits = circuit.num_qubits
    probs = Statevector.from_instruction(circuit).probabilities()
    values = np.fromiter(
        (value_map(format(i, f"0{n_qubits}b")) for i in range(len(probs))),
        dtype=float,
        count=len(probs),
    )
    rng = np.random.default_rng(seed)

    def sample_many(n: int) -> list[float]:
        idx = rng.choice(len(probs), size=int(n), p=probs)
        return values[idx].tolist()

    return sample_many


# ── Noisy simulation — harvest exp2/exp4 make_zz_sampler ───────────────────────

def noise_model_sampler(
    circuit: QuantumCircuit,
    outcome_map: OutcomeMap,
    noise_model: NoiseModel,
    *,
    seed: int,
    buffer_size: int = 50_000,
) -> SampleMany:
    """Buffered AerSimulator sampler under a noise model.

    Pre-runs ``buffer_size`` shots once and serves shuffled slices via a cursor,
    so repeated draws across EBS checkpoints are independent rather than
    re-simulated with a reused seed. On buffer exhaustion it falls back to
    Bernoulli draws matched to the buffer's empirical mean.

    Args:
        circuit: Circuit to simulate (``measure_all`` added if unmeasured).
        outcome_map: Bitstring → ±1 eigenvalue for the target observable.
        noise_model: Qiskit Aer noise model applied to the simulation.
        seed: Seed for the simulator and the shuffle/fallback generator.
        buffer_size: Number of shots pre-simulated into the pool.

    Returns:
        A ``SampleMany`` drawing ±1 outcomes from the buffered pool.
    """
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    qc_t = transpile(_ensure_measured(circuit), sim, optimization_level=0)
    counts = sim.run(qc_t, shots=int(buffer_size)).result().get_counts()

    buffer = np.asarray(_counts_to_samples(counts, outcome_map), dtype=float)
    rng = np.random.default_rng(seed)
    rng.shuffle(buffer)
    cursor = [0]

    def sample_many(n: int) -> list[float]:
        n = int(n)
        start, end = cursor[0], cursor[0] + n
        if end > len(buffer):
            p = (1.0 + float(buffer.mean())) / 2.0
            cursor[0] = 0
            return (2.0 * rng.binomial(1, p, size=n) - 1.0).tolist()
        cursor[0] = end
        return buffer[start:end].tolist()

    return sample_many


# ── Online hardware (SamplerV2) — harvest exp_qce1 make_backend_sampler_* ──────

def _is_ibm_backend(backend: object) -> bool:
    """True if ``backend`` requires the SamplerV2 primitive (real IBM hardware)."""
    try:
        from qiskit_ibm_runtime import IBMBackend
    except ImportError:
        return False
    return isinstance(backend, IBMBackend)


def _counts_from_pub(pub: object) -> dict[str, int]:
    """Extract merged classical-register counts from a SamplerV2 pub result."""
    if hasattr(pub, "join_data"):
        joined = pub.join_data()
        if joined is not None:
            return joined.get_counts()
    data = pub.data  # type: ignore[attr-defined]
    for name in data:
        return getattr(data, name).get_counts()
    raise ValueError("SamplerV2 pub result has no classical-register data")


def _submit(
    backend: object,
    qc_t: QuantumCircuit,
    shots: int,
    tags: Optional[dict],
) -> tuple[dict[str, int], str]:
    """Submit one job and return ``(counts, job_id)`` captured at submission."""
    if _is_ibm_backend(backend):
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(backend)
        if tags:
            sampler.options.environment.job_tags = [f"{k}:{v}" for k, v in tags.items()]
        job = sampler.run([(qc_t,)], shots=shots)
        job_id = job.job_id()
        return _counts_from_pub(job.result()[0]), job_id

    job = backend.run(qc_t, shots=shots)  # type: ignore[attr-defined]
    return job.result().get_counts(), job.job_id()


def backend_sampler(
    circuit: QuantumCircuit,
    outcome_map: OutcomeMap,
    backend: object,
    *,
    tags: Optional[dict] = None,
    seed: Optional[int] = None,
) -> SampleMany:
    """Online sampler submitting each draw as a backend job.

    Uses SamplerV2 for real IBM backends and ``backend.run`` for AerSimulator
    (dry-run), transpiling once. Every submission's ``job_id`` is recorded on
    the returned callable's ``.job_ids`` attribute so provenance harnesses
    (``backends/ibm.py``) can pair reported numbers to raw jobs.

    Args:
        circuit: Circuit with measurements defining the readout.
        outcome_map: Bitstring → ±1 eigenvalue for the target observable.
        backend: IBM ``IBMBackend`` or an ``AerSimulator``-like backend.
        tags: Optional job tags attached to each IBM submission (``k:v`` form).
        seed: Seed for shuffling returned outcomes (order only; independence
            comes from distinct jobs). ``None`` leaves ordering unshuffled.

    Returns:
        A ``SampleMany`` with a ``job_ids: list[str]`` attribute.
    """
    qc_t = transpile(circuit, backend, optimization_level=3)
    rng = np.random.default_rng(seed)
    job_ids: list[str] = []

    def sample_many(n: int) -> list[float]:
        counts, job_id = _submit(backend, qc_t, int(n), tags)
        job_ids.append(job_id)
        samples = np.asarray(_counts_to_samples(counts, outcome_map), dtype=float)
        if seed is not None:
            rng.shuffle(samples)
        return samples.tolist()

    sample_many.job_ids = job_ids  # type: ignore[attr-defined]
    return sample_many


# ── Offline replay — harvest exp_qce3_hw make_offline_sampler ──────────────────

def offline_replay_sampler(outcomes: Sequence[float]) -> SampleMany:
    """Replay pre-collected ±1 outcomes via a consuming cursor.

    Re-runs EBS/Hoeffding on hardware counts already fetched (free — no compute
    minutes). Each draw returns the next slice; the buffer is consumed once.

    Args:
        outcomes: Pre-collected ±1 eigenvalues (already shuffled upstream).

    Returns:
        A ``SampleMany`` serving successive slices of ``outcomes``.
    """
    buffer = np.asarray(outcomes, dtype=float)
    cursor = [0]

    def sample_many(n: int) -> list[float]:
        start, end = cursor[0], cursor[0] + int(n)
        cursor[0] = end
        return buffer[start:end].tolist()

    return sample_many
