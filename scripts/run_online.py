"""Real-time online EBS stopping via job chaining (the OnlineStopper, G10).

Instead of one full-budget job, this submits shots batch-by-batch at the
geometric checkpoints and STOPS submitting once the empirical Bernstein radius
drops below epsilon -- so it *realizes* the saving (fewer shots actually run),
not just the counterfactual one. On IBM the batches run inside a Session so they
execute back-to-back without re-queuing; ``--dry-run`` swaps in AerSimulator to
exercise the identical loop with zero QPU.

    python3 scripts/run_online.py --dry-run                       # Aer mock
    python3 scripts/run_online.py --workload swap --beta 1.7      # real ibm_fez
"""

from __future__ import annotations

import argparse
import json
import os
from math import cos

import numpy as np
from dotenv import load_dotenv
from qiskit import transpile

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    ideal_zz,
    qaoa_maxcut_circuit,
    swap_test_1qubit,
    zz_outcome_map,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, DELTA, SEED = 0.02, 0.01, 42
THETA1, THETA2 = 0.3, 0.8
GAMMA, BETA_Q = 0.783, 0.438


def _ancilla_outcome(bitstring: str) -> float:
    return 1.0 if bitstring[-1] == "0" else -1.0


def _counts_to_samples(counts, outcome_map):
    out = []
    for bits, c in counts.items():
        out.extend([outcome_map(bits.replace(" ", ""))] * c)
    return out


def _pub_counts(result):
    pub = result[0]
    if hasattr(pub, "join_data") and pub.join_data() is not None:
        return pub.join_data().get_counts()
    data = pub.data
    for name in data:
        return getattr(data, name).get_counts()
    raise ValueError("no classical data in pub")


def _workload(name):
    if name == "swap":
        qc = swap_test_1qubit(THETA1, THETA2); qc.measure_all()
        return qc, _ancilla_outcome, cos((THETA1 - THETA2) / 2.0) ** 2
    qc = qaoa_maxcut_circuit(GAMMA, BETA_Q); qc.measure_all()
    return qc, zz_outcome_map, ideal_zz(GAMMA, BETA_Q)


def _make_sampler(qc_t, omap, submit, out_dir, tags):
    """Return (sample_many, job_ids): each call submits one batch and files it."""
    job_ids, jobs_dir = [], os.path.join(out_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    def sample_many(n):
        counts, jid = submit(qc_t, int(n))
        job_ids.append(jid)
        with open(os.path.join(jobs_dir, f"{jid}.json"), "w") as f:
            json.dump({"job_id": jid, "shots": int(n), "tags": tags, "counts": counts}, f, default=str)
        return _counts_to_samples(counts, omap)

    return sample_many, job_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--job-mode", action="store_true",
                    help="open-plan route: each batch is an independent QUEUED job (no Session)")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--workload", default="swap", choices=["swap", "qaoa"])
    ap.add_argument("--beta", type=float, default=1.1,
                    help="checkpoint ratio; larger => fewer batches (fewer jobs)")
    ap.add_argument("--n-min", type=int, default=10,
                    help="first checkpoint / min shots before checking; raise to skip tiny early batches")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "online"))
    args = ap.parse_args()

    qc, omap, ideal = _workload(args.workload)
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    stopper = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0, beta=args.beta)
    n_ck = len(stopper.checkpoints())
    tags = {"workload": args.workload, "method": "online-ebs", "eps": EPS, "delta": DELTA, "beta": args.beta}
    mode = "DRY RUN/Aer" if args.dry_run else ("REAL ibm_fez JOB-MODE/queue" if args.job_mode
                                               else "REAL ibm_fez SESSION")
    print(f"Online EBS: workload={args.workload} beta={args.beta} checkpoints={n_ck} n_H={n_h} ({mode})")

    if args.dry_run:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        qc_t = transpile(qc, backend, optimization_level=1)
        _rng = np.random.default_rng(SEED)

        def submit(circ, n):
            s = int(_rng.integers(0, 2**31))
            counts = backend.run(circ, shots=n, seed_simulator=s).result().get_counts()
            return counts, f"aer-{s}"

        sm, jids = _make_sampler(qc_t, omap, submit, args.out, tags)
        res = stopper.run_batched(sm)
    elif args.job_mode:
        load_dotenv(os.path.join(HERE, "..", "examples", "pcsc2026", ".env"))
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])
        backend = svc.backend(args.backend)
        qc_t = transpile(qc, backend, optimization_level=1)
        sampler = SamplerV2(mode=backend)  # each run() = one independent QUEUED job

        def submit(circ, n):
            job = sampler.run([(circ,)], shots=n)
            print(f"    [batch {len(jids) + 1}] job {job.job_id()} submitted ({n} shots), waiting in queue...", flush=True)
            counts = _pub_counts(job.result())
            print(f"    [batch {len(jids) + 1}] done.", flush=True)
            return counts, job.job_id()

        sm, jids = _make_sampler(qc_t, omap, submit, args.out, tags)
        res = stopper.run_batched(sm)
    else:
        load_dotenv(os.path.join(HERE, "..", "examples", "pcsc2026", ".env"))
        from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])
        backend = svc.backend(args.backend)
        qc_t = transpile(qc, backend, optimization_level=1)
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)

            def submit(circ, n):
                job = sampler.run([(circ,)], shots=n)
                return _pub_counts(job.result()), job.job_id()

            sm, jids = _make_sampler(qc_t, omap, submit, args.out, tags)
            res = stopper.run_batched(sm)

    reduction = n_h / res.n if res.n else float("nan")
    print(f"  REALIZED: shots actually submitted = {res.n}  (vs n_H={n_h}, {reduction:.2f}x saved)")
    print(f"  batches submitted = {len(jids)}  stopped_by={res.stopped_by}")
    print(f"  estimate={res.estimate:+.4f}  |err vs ideal|={abs(res.estimate - ideal):.4f}")
    print(f"  job_ids: {jids if args.dry_run else ', '.join(jids)}")


if __name__ == "__main__":
    main()
