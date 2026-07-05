"""Hardware runs on ibm_fez with by-construction provenance (Table 5.3, Appendix C).

Each workload submits ONE tagged job at the Hoeffding budget via
``run_and_record`` (job_id captured at submission, raw counts JSON filed), then
applies EBS-geom post hoc to the returned shot record. Pairing is automatic: the
job_id, tags, and counts all come from one handle. Use ``--dry-run`` to exercise
the whole path on AerSimulator (no QPU) before spending device time.

    python3 scripts/run_hardware.py --dry-run            # Aer, no QPU
    python3 scripts/run_hardware.py --workloads swap --shots 26492   # real
"""

from __future__ import annotations

import argparse
import os
from math import cos

import numpy as np
from dotenv import load_dotenv
from qiskit import transpile

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    H2_ANGLES,
    HoeffdingPlanner,
    ideal_zz,
    offline_replay_sampler,
    qaoa_maxcut_circuit,
    swap_test_1qubit,
    vqe_ansatz,
    zz_outcome_map,
)
from qamp_shotplanner.backends.ibm import run_and_record, snapshot_calibration, write_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, DELTA, SEED = 0.02, 0.01, 42
THETA1, THETA2 = 0.3, 0.8
GAMMA, BETA = 0.783, 0.438


def _ancilla_outcome(bitstring: str) -> float:
    return 1.0 if bitstring[-1] == "0" else -1.0


def _workloads():
    swap = swap_test_1qubit(THETA1, THETA2)
    swap.measure_all()
    qaoa = qaoa_maxcut_circuit(GAMMA, BETA)
    qaoa.measure_all()
    f_ideal = cos((THETA1 - THETA2) / 2.0) ** 2
    vqe = vqe_ansatz(*H2_ANGLES[0.74])  # Z-basis; raw counts carry IZ/ZI/ZZ for offline energy
    vqe.measure_all()
    return {
        "swap": (swap, _ancilla_outcome, f_ideal),
        "qaoa": (qaoa, zz_outcome_map, ideal_zz(GAMMA, BETA)),
        "vqe": (vqe, zz_outcome_map, 0.0),  # ideal placeholder; real energy computed offline
    }


def _get_backend(dry_run: bool, name: str):
    if dry_run:
        from qiskit_aer import AerSimulator
        return AerSimulator()
    from qiskit_ibm_runtime import QiskitRuntimeService
    load_dotenv(os.path.join(HERE, "..", "examples", "pcsc2026", ".env"))
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])
    return svc.backend(name) if name else svc.least_busy(operational=True, simulator=False)


def _counts_to_samples(counts, outcome_map):
    out = []
    for bits, c in counts.items():
        out.extend([outcome_map(bits.replace(" ", ""))] * c)
    return out


def _post_hoc_ebs(samples, seed):
    rng = np.random.default_rng(seed)
    buf = np.asarray(samples, dtype=float)
    rng.shuffle(buf)
    return EmpiricalBernsteinStopper(
        epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
    ).run_batched(offline_replay_sampler(buf.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="AerSimulator, no QPU")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--workloads", nargs="+", default=["swap", "qaoa"])
    ap.add_argument("--shots", type=int, default=None, help="override (default n_H)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "hardware"))
    args = ap.parse_args()

    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    shots = args.shots or n_h
    backend = _get_backend(args.dry_run, args.backend)
    bname = getattr(backend, "name", "aer")
    print(f"Backend: {bname}  shots/job: {shots}  workloads: {args.workloads}  "
          f"{'(DRY RUN)' if args.dry_run else '(REAL QPU)'}")
    os.makedirs(args.out, exist_ok=True)

    wl = _workloads()
    for name in args.workloads:
        circuit, omap, ideal = wl[name]
        qc_t = transpile(circuit, backend, optimization_level=1)
        tags = {"workload": name, "method": "hoeffding-then-ebs", "eps": EPS,
                "delta": DELTA, "seed": SEED, "shots": shots}
        print(f"  submitting {name} ({shots} shots)...")
        counts, rec = run_and_record(qc_t, omap, backend, tags=tags, out_dir=args.out)
        samples = _counts_to_samples(counts, omap)
        mu_hoeff = float(np.mean(samples))
        r = _post_hoc_ebs(samples, SEED)
        print(f"    job_id={rec.job_ids[0]}")
        print(f"    Hoeffding: est={mu_hoeff:+.4f} n={len(samples)} |err vs ideal|={abs(mu_hoeff-ideal):.4f}")
        print(f"    EBS-geom : est={r.estimate:+.4f} n={r.n} ({len(samples)/r.n:.2f}x) "
              f"stopped_by={r.stopped_by} |err|={abs(r.estimate-ideal):.4f}")

    snapshot_calibration(backend, args.out)
    write_manifest(args.out, script="run_hardware.py", git_sha="", seed=SEED,
                   eps=EPS, delta=DELTA, timestamp="")
    print(f"  wrote bundle to {args.out}")


if __name__ == "__main__":
    main()
