"""A real ZNE point on hardware via unitary gate folding (companion to the sim sweep).

On hardware the noise level is fixed, so ZNE amplifies it by GLOBAL unitary
folding: the circuit unitary U is replaced by U (U^dagger U)^k, giving an
effective noise scale lambda = 2k+1 (k=0,1,2 -> lambda=1,3,5). The folded
circuits are run on ibm_fez, the ancilla observable is linearly extrapolated to
lambda=0, and the empirical Bernstein rule is applied post hoc to measure the
overhead gamma^2 = tau_mit/tau_unmit and the removed bias -- the same diagnostic
as the simulation sweep, now at the device's native noise. Use --dry-run to
validate the folding on an Aer noise model before spending QPU.

    python3 scripts/run_hardware_zne.py --dry-run          # Aer noise model
    python3 scripts/run_hardware_zne.py --shots 8000       # real ibm_fez
"""

from __future__ import annotations

import argparse
import json
import os
from math import cos

import numpy as np
from dotenv import load_dotenv
from qiskit import transpile

from qamp_shotplanner import EmpiricalBernsteinStopper, offline_replay_sampler, swap_test_1qubit

EPS, DELTA, SEED = 0.02, 0.01, 42
THETA1, THETA2 = 0.3, 0.8
F_IDEAL = cos((THETA1 - THETA2) / 2.0) ** 2
LAMBDAS = [1, 3, 5]
ZNE_W = np.array([13.0 / 12.0, 1.0 / 3.0, -5.0 / 12.0])   # linear LSQ extrapolation of [1,3,5] to 0
R_ZNE = 2.0 * float(np.sum(np.abs(ZNE_W)))
HERE = os.path.dirname(os.path.abspath(__file__))


def fold(u, k):
    """Global unitary folding: U (U^dagger U)^k, noise scale lambda = 2k+1.

    Barriers between the folds stop the transpiler from cancelling U^dagger U to
    the identity, which is what preserves the noise amplification.
    """
    folded = u.copy()
    uinv = u.inverse()
    for _ in range(k):
        folded.barrier()
        folded = folded.compose(uinv)
        folded.barrier()
        folded = folded.compose(u)
    return folded


def _ancilla(bits):
    return 1.0 if bits.replace(" ", "")[-1] == "0" else -1.0


def _samples(counts):
    out = []
    for bits, c in counts.items():
        out.extend([_ancilla(bits)] * c)
    return np.asarray(out, dtype=float)


def _post_hoc_ebs(samples, a, b, seed):
    buf = samples.copy()
    np.random.default_rng(seed).shuffle(buf)
    return EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=a, b=b).run_batched(
        offline_replay_sampler(buf.tolist()))


def _aer_noise(p=0.005):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h", "rx", "ry", "rz", "x", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(min(5 * p, 1), 2), ["cx", "cz"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return AerSimulator(noise_model=nm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "hardware_zne"))
    args = ap.parse_args()

    u = swap_test_1qubit(THETA1, THETA2)               # unitary part, no measurement
    circuits = {}
    for lam, k in zip(LAMBDAS, [0, 1, 2]):
        qc = fold(u, k)
        qc.measure_all()
        circuits[lam] = qc

    if args.dry_run:
        backend = _aer_noise()
        submit = lambda circ, n: (backend.run(transpile(circ, backend, optimization_level=1),
                                              shots=n, seed_simulator=SEED).result().get_counts(), "aer")
        bname = "aer-noise"
    else:
        load_dotenv(os.path.join(HERE, "..", "examples", "pcsc2026", ".env"))
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])
        backend = svc.backend(args.backend)
        sampler = SamplerV2(mode=backend)

        def submit(circ, n):
            job = sampler.run([(transpile(circ, backend, optimization_level=1),)], shots=n)
            res = job.result()[0]
            counts = (res.join_data().get_counts() if hasattr(res, "join_data") and res.join_data() is not None
                      else next(getattr(res.data, nm).get_counts() for nm in res.data))
            return counts, job.job_id()
        bname = args.backend

    os.makedirs(os.path.join(args.out, "jobs"), exist_ok=True)
    print(f"Hardware ZNE on {bname}: SWAP F_ideal={F_IDEAL:.4f}, folding lambda={LAMBDAS}, "
          f"shots/level={args.shots}  {'(DRY RUN)' if args.dry_run else '(REAL QPU)'}")
    streams, evs, jobids = {}, {}, []
    for lam in LAMBDAS:
        counts, jid = submit(circuits[lam], args.shots)
        jobids.append(jid)
        with open(os.path.join(args.out, "jobs", f"lam{lam}_{jid}.json"), "w") as f:
            json.dump({"lambda": lam, "job_id": jid, "shots": args.shots, "counts": counts}, f)
        s = _samples(counts)
        streams[lam] = s
        evs[lam] = float(np.mean(s))
        print(f"  lambda={lam}: <Z>={evs[lam]:+.4f}  (depth-folded, job {jid})")

    mu_dev = evs[1]
    mu_zne = float(np.dot(ZNE_W, [evs[1], evs[3], evs[5]]))
    m = min(len(streams[l]) for l in LAMBDAS)
    y = ZNE_W[0] * streams[1][:m] + ZNE_W[1] * streams[3][:m] + ZNE_W[2] * streams[5][:m]
    r_un = _post_hoc_ebs(streams[1], -1.0, 1.0, SEED)
    r_zne = _post_hoc_ebs(y, -R_ZNE / 2, R_ZNE / 2, SEED + 1)
    tau_un, tau_zne = r_un.n, 3 * r_zne.n
    print(f"  unmitigated : mu_dev={mu_dev:+.4f}  bias={abs(mu_dev - F_IDEAL):.4f}  tau={tau_un}")
    print(f"  ZNE (l=0)   : mu_zne={mu_zne:+.4f}  bias={abs(mu_zne - F_IDEAL):.4f}  tau={tau_zne}")
    print(f"  measured overhead gamma^2 = tau_zne/tau_un = {tau_zne / tau_un:.2f}x   job_ids={jobids}")


if __name__ == "__main__":
    main()
