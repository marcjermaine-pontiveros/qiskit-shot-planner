"""IBM Quantum provenance harness: tagged submission, raw-result capture, and
calibration/manifest snapshots for reproducibility.

Every reported number is re-derived from raw counts. Pairing between a job and
its result is by construction: `run_and_record` captures ``job.job_id()`` from
the same handle it calls ``.result()`` on, and files the raw JSON under that id.
Provenance (workload/method/eps/delta/seed) travels inside the job's ``tags``,
never a manual side table.

All ``qiskit_ibm_runtime`` imports are guarded so this module imports cleanly in
CI without the package or credentials, and runs against an ``AerSimulator``
backend for dry runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from qamp_shotplanner.planners.empirical_bernstein import eb_radius_maurer

OutcomeMap = Callable[[str], float]

_R = 2.0  # range b - a for ±1 eigenvalue outcomes


@dataclass
class RunRecord:
    """Provenance record for one estimation run, re-derived from raw counts.

    Attributes:
        workload: Workload identifier (e.g. "qaoa", "vqe_h2").
        method: Stopping rule — "hoeffding" | "ebs" | "anytime".
        eps: Statistical tolerance ε.
        delta: Total failure probability δ.
        seed: RNG seed pinning the run.
        backend: Backend name the job ran on.
        job_ids: Submission job ids backing this record.
        shots: Total shots re-derived from raw counts.
        estimate: Mean estimate re-derived from raw counts.
        radius: Empirical Bernstein radius at the reported n.
        stopped_by: What ended sampling (e.g. the method or "cap").
        timestamp: Wall-clock timestamp (passed in, not generated here).
        git_sha: Source revision (passed in, not generated here).
    """

    workload: str
    method: str
    eps: float
    delta: float
    seed: int
    backend: str
    job_ids: list[str]
    shots: int
    estimate: float
    radius: float
    stopped_by: str
    timestamp: str
    git_sha: str


# ── Backend dispatch ──────────────────────────────────────────────────────────

def _is_ibm_backend(backend: Any) -> bool:
    """True if the backend needs SamplerV2 (real IBM hardware/runtime)."""
    try:
        from qiskit_ibm_runtime import IBMBackend
    except ImportError:
        return False
    return isinstance(backend, IBMBackend)


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return name() if callable(name) else (name or str(backend))


def _tag_list(tags: dict) -> list[str]:
    """Flatten a provenance dict into ``key=value`` job tags (server-side)."""
    return [f"{k}={v}" for k, v in tags.items()]


def _submit(circuit: Any, backend: Any, shots: int, job_tags: list[str]) -> Any:
    """Submit one shots-fixed job; attach job_tags on the IBM path.

    Circuit is assumed backend-ready (transpile in the caller / sampler factory).
    """
    if _is_ibm_backend(backend):
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
        try:  # tag placement varies across runtime versions — best effort
            sampler.options.environment.job_tags = job_tags
        except Exception:
            pass
        return sampler.run([(circuit,)], shots=int(shots))
    return backend.run(circuit, shots=int(shots))


def _counts_from_result(result: Any) -> dict[str, int]:
    """Extract a ``{bitstring: count}`` dict across Aer and SamplerV2 results."""
    if hasattr(result, "get_counts"):  # AerSimulator / BackendV2.run
        return {str(k): int(v) for k, v in result.get_counts().items()}
    pub = result[0]  # SamplerV2 PrimitiveResult, single PUB
    if hasattr(pub, "join_data"):
        return {str(k): int(v) for k, v in pub.join_data().get_counts().items()}
    data = pub.data
    for name in getattr(data, "keys", list)():
        creg = getattr(data, name)
        if hasattr(creg, "get_counts"):
            return {str(k): int(v) for k, v in creg.get_counts().items()}
    raise ValueError("could not extract counts from result")


def _derive(counts: dict[str, int], outcome_map: OutcomeMap) -> tuple[float, float, int]:
    """Re-derive (estimate, biased variance, shots) from raw counts."""
    shots = sum(counts.values())
    if shots == 0:
        return 0.0, 0.0, 0
    mean = sum(c * outcome_map(b) for b, c in counts.items()) / shots
    var = sum(c * (outcome_map(b) - mean) ** 2 for b, c in counts.items()) / shots
    return mean, var, shots


# ── Public API ────────────────────────────────────────────────────────────────

def run_and_record(
    circuit: Any,
    outcome_map: OutcomeMap,
    backend: Any,
    *,
    tags: dict,
    out_dir: str | os.PathLike,
) -> tuple[dict[str, int], RunRecord]:
    """Submit one tagged job, capture its id, and file the raw result JSON.

    Pairing is by construction: the job id is read from the same handle whose
    ``.result()`` produced ``counts``, and the raw JSON is written to
    ``out_dir/jobs/<job_id>.json``. The estimate, shots, and radius on the
    returned record are re-derived from those raw counts.

    Args:
        circuit: Backend-ready measured circuit (already transpiled if on IBM).
        outcome_map: Bitstring -> ±1 eigenvalue for the observable.
        backend: IBM ``IBMBackend`` or an ``AerSimulator`` for dry runs.
        tags: Provenance carried into the job's server-side tags. Recognized
            keys: workload, method, eps, delta, seed, shots, timestamp, git_sha.
        out_dir: Output root; raw job JSON lands in ``out_dir/jobs/``.

    Returns:
        A ``(counts, RunRecord)`` pair.
    """
    shots = int(tags["shots"])
    job = _submit(circuit, backend, shots, _tag_list(tags))
    job_id = job.job_id()
    result = job.result()
    counts = _counts_from_result(result)

    estimate, var, n = _derive(counts, outcome_map)
    delta = float(tags["delta"])
    method = str(tags["method"])
    radius = eb_radius_maurer(n=n, R=_R, var_biased=var, delta=delta)

    jobs_dir = os.path.join(out_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    raw = {
        "job_id": job_id,
        "backend": _backend_name(backend),
        "tags": tags,
        "shots": n,
        "counts": counts,
    }
    with open(os.path.join(jobs_dir, f"{job_id}.json"), "w") as f:
        json.dump(raw, f, indent=2, default=str)

    record = RunRecord(
        workload=str(tags["workload"]),
        method=method,
        eps=float(tags["eps"]),
        delta=delta,
        seed=int(tags["seed"]),
        backend=_backend_name(backend),
        job_ids=[job_id],
        shots=n,
        estimate=estimate,
        radius=radius,
        stopped_by=str(tags.get("stopped_by", method)),
        timestamp=str(tags.get("timestamp", "")),
        git_sha=str(tags.get("git_sha", "")),
    )
    return counts, record


def fetch_job(service: Any, job_id: str) -> dict[str, Any]:
    """Retrieve a past job's raw result as a JSON-safe counts dict.

    This is FREE — it reads stored results and consumes no compute minutes.

    Args:
        service: A ``QiskitRuntimeService`` instance.
        job_id: The submission id to fetch.

    Returns:
        ``{"job_id", "backend", "shots", "counts"}`` re-derived from the result.
    """
    job = service.job(job_id)
    counts = _counts_from_result(job.result())
    return {
        "job_id": job_id,
        "backend": _backend_name(getattr(job, "backend", lambda: None)() or ""),
        "shots": sum(counts.values()),
        "counts": counts,
    }


def snapshot_calibration(backend: Any, out_dir: str | os.PathLike) -> dict[str, Any]:
    """Snapshot T1/T2, readout, and two-qubit gate errors to calibration.json.

    Works from ``backend.properties()`` (IBM); returns a minimal record when a
    backend exposes no properties (e.g. ideal AerSimulator).

    Args:
        backend: The backend to snapshot.
        out_dir: Directory to write ``calibration.json`` into.

    Returns:
        The snapshot dict that was written.
    """
    props = getattr(backend, "properties", lambda: None)()
    n = getattr(backend, "num_qubits", 0)
    snap: dict[str, Any] = {
        "backend": _backend_name(backend),
        "num_qubits": n,
        "t1_us": {},
        "t2_us": {},
        "readout_error": {},
        "gate_errors": {},
    }
    if props is not None:
        snap["last_update"] = str(getattr(props, "last_update_date", ""))
        for q in range(n):
            for key, fn, scale in (
                ("t1_us", props.t1, 1e6),
                ("t2_us", props.t2, 1e6),
                ("readout_error", props.readout_error, 1.0),
            ):
                try:
                    v = fn(q)
                except Exception:
                    v = None
                if v is not None:
                    snap[key][str(q)] = v * scale
        for gate in getattr(props, "gates", []):
            if gate.gate in ("cx", "ecr", "cz"):
                label = f"{gate.gate}_" + "_".join(str(q) for q in gate.qubits)
                for p in gate.parameters:
                    if p.name == "gate_error":
                        snap["gate_errors"][label] = p.value

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(snap, f, indent=2, default=str)
    return snap


def write_manifest(
    out_dir: str | os.PathLike,
    *,
    script: str,
    git_sha: str,
    seed: int,
    eps: float,
    delta: float,
    timestamp: str,
) -> dict[str, Any]:
    """Write a run manifest (``manifest.json``) tying outputs to their source.

    Args:
        out_dir: Directory to write ``manifest.json`` into.
        script: Driver script path/name that produced this run.
        git_sha: Source revision.
        seed: RNG seed.
        eps: Statistical tolerance ε.
        delta: Failure probability δ.
        timestamp: Wall-clock timestamp.

    Returns:
        The manifest dict that was written.
    """
    manifest = {
        "script": script,
        "git_sha": git_sha,
        "seed": seed,
        "eps": eps,
        "delta": delta,
        "timestamp": timestamp,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
