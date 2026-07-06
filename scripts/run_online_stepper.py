"""Resumable, revisitable online EBS on the open plan (job-mode queue).

Real-time online stopping needs a Session (paid). On the open plan we get the
same *adaptive* behaviour by chaining independent QUEUED jobs -- but a job can
sit in the queue for a long time, so we never want to babysit a live process.

This driver does ONE step per invocation and persists everything to a state
file. Because IBM retains job results by ID indefinitely, the run is fully
resumable: submit batch -> exit; come back later -> it checks the pending job,
accumulates, and either stops or submits the next batch. All job IDs accumulate
in state.json. Works for any bounded observable: state carries (n, sum_x,
sumsq_x), so the per-shot value may be a general number in [-1, 1] (e.g. the
utility-scale magnetization) rather than only +/-1.

    # SWAP fidelity (ancilla +/-1):
    python3 scripts/run_online_stepper.py --workload swap --n-min 2000 --beta 1.35
    # utility-scale TFIM magnetization (dry-run on Aer first, then hardware):
    python3 scripts/run_online_stepper.py --workload tfim --n-qubits 20 --steps 10 --dry-run
    python3 scripts/run_online_stepper.py --workload tfim --n-qubits 20 --steps 10
    # revisit any run to advance one step:
    python3 scripts/run_online_stepper.py --state results/online/tfim_stepper
"""

from __future__ import annotations

import argparse
import json
import os
from math import cos

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    qaoa_maxcut_circuit,
    swap_test_1qubit,
    zz_outcome_map,
)
from qamp_shotplanner.planners.empirical_bernstein import eb_radius_maurer

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, DELTA = 0.02, 0.01
THETA1, THETA2 = 0.3, 0.8
GAMMA, BETA_Q = 0.783, 0.438
TFIM_J, TFIM_H, TFIM_DT = 1.0, -5.0, 0.1


# ---- workloads: each returns (circuit_with_measure, per_shot_value_fn) ----

def _tfim_circuit(n_qubits, n_steps):
    zz = [("ZZ", [i, i + 1], -TFIM_J) for i in range(n_qubits - 1)]
    x = [("X", [i], -TFIM_H) for i in range(n_qubits)]
    ham = SparsePauliOp.from_sparse_list([*zz, *x], num_qubits=n_qubits).simplify()
    step = PauliEvolutionGate(ham, TFIM_DT, synthesis=LieTrotter())
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        qc.append(step, qc.qubits)
    qc.measure_all()
    return qc


def _workload(st):
    name = st["workload"]
    if name == "swap":
        qc = swap_test_1qubit(THETA1, THETA2); qc.measure_all()
        return qc, lambda b: 1.0 if b[-1] == "0" else -1.0
    if name == "qaoa":
        qc = qaoa_maxcut_circuit(GAMMA, BETA_Q); qc.measure_all()
        return qc, lambda b: zz_outcome_map(b)
    if name == "tfim":
        nq = st["n_qubits"]
        qc = _tfim_circuit(nq, st["steps"])
        return qc, lambda b: (nq - 2 * b.count("1")) / nq
    raise ValueError(name)


def _accumulate(counts, value_fn):
    n = sx = sxx = 0.0
    for bits, c in counts.items():
        v = value_fn(bits.replace(" ", ""))
        n += c
        sx += v * c
        sxx += v * v * c
    return int(n), sx, sxx


def _pub_counts(result):
    pub = result[0]
    if hasattr(pub, "join_data") and pub.join_data() is not None:
        return pub.join_data().get_counts()
    for name in pub.data:
        return getattr(pub.data, name).get_counts()
    raise ValueError("no classical data in pub")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)


def _service():
    load_dotenv(os.path.join(HERE, "..", "examples", "pcsc2026", ".env"))
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"])


def _radius(st):
    n = st["n_accum"]
    mean = st["sum_x"] / n
    var = max(0.0, st["sumsq_x"] / n - mean * mean)
    return eb_radius_maurer(n=n, R=st["R"], var_biased=var, delta=st["deltas"][st["cur_idx"]]), mean


def _fold(st, counts, value_fn):
    n, sx, sxx = _accumulate(counts, value_fn)
    st["n_accum"] += n
    st["sum_x"] += sx
    st["sumsq_x"] += sxx
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=None)
    ap.add_argument("--workload", default="swap", choices=["swap", "qaoa", "tfim"])
    ap.add_argument("--n-qubits", type=int, default=20, help="tfim only")
    ap.add_argument("--steps", type=int, default=10, help="tfim Trotter steps")
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--beta", type=float, default=1.35)
    ap.add_argument("--n-min", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true",
                    help="run the whole chain on AerSimulator (no QPU), for validation")
    args = ap.parse_args()

    default_dir = f"{args.workload}_stepper" + ("" if args.workload != "tfim"
                                                else f"_n{args.n_qubits}_s{args.steps}")
    state_dir = args.state or os.path.join(HERE, "..", "results", "online", default_dir)
    os.makedirs(os.path.join(state_dir, "jobs"), exist_ok=True)
    spath = os.path.join(state_dir, "state.json")

    if not os.path.exists(spath):
        n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
        stp = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0,
                                        beta=args.beta, n_min=args.n_min)
        st = {
            "workload": args.workload, "n_qubits": args.n_qubits, "steps": args.steps,
            "backend": args.backend, "eps": EPS, "delta": DELTA,
            "beta": args.beta, "n_min": args.n_min, "n_h": n_h,
            "checkpoints": stp.checkpoints(), "deltas": list(stp._deltas),
            "R": stp.R, "alpha": stp.alpha, "epsilon_stat": stp.epsilon_stat,
            "cur_idx": 0, "n_accum": 0, "sum_x": 0.0, "sumsq_x": 0.0,
            "pending_job_id": None, "pending_shots": 0,
            "job_ids": [], "status": "running", "result": None,
        }
        _save(spath, st)
        tag = f"{args.workload}" + (f" N={args.n_qubits} steps={args.steps}" if args.workload == "tfim" else "")
        print(f"[init] {tag}: {len(st['checkpoints'])} checkpoints {st['checkpoints']}  n_H={n_h}")
    else:
        st = _load(spath)

    if st["status"] != "running":
        print(f"[done] status={st['status']}  result={st['result']}")
        print(f"       job_ids: {[j['job_id'] for j in st['job_ids']]}")
        return

    qc, value_fn = _workload(st)

    # ---- DRY RUN: run the whole chain on Aer, no persistence semantics needed ----
    if args.dry_run:
        from qiskit_aer import AerSimulator
        import numpy as np
        backend = AerSimulator()
        qc_t = transpile(qc, basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=1)
        rng = np.random.default_rng(42)
        prev = 0
        for k, cp in enumerate(st["checkpoints"]):
            dn = cp - prev
            counts = backend.run(qc_t, shots=dn, seed_simulator=int(rng.integers(0, 2**31))).result().get_counts()
            _fold(st, counts, value_fn)
            prev = cp
            st["cur_idx"] = k
            rad, mean = _radius(st)
            print(f"[dry k={k}] n={st['n_accum']} mean={mean:+.4f} radius={rad:.5f}")
            if rad < st["epsilon_stat"]:
                print(f"[DRY STOP] n={st['n_accum']} ({st['n_h']/st['n_accum']:.2f}x)")
                return
        print(f"[DRY CAP] n={st['n_accum']}")
        return

    svc = _service()
    backend = svc.backend(st["backend"])

    # ---- 1) resolve pending job ----
    if st["pending_job_id"]:
        job = svc.job(st["pending_job_id"])
        s = str(job.status())
        if "DONE" not in s.upper():
            print(f"[wait] idx={st['cur_idx']} job {st['pending_job_id']} status={s} "
                  f"(pending ahead: {backend.status().pending_jobs}). Re-run later.")
            return
        counts = _pub_counts(job.result())
        with open(os.path.join(state_dir, "jobs", f"{st['pending_job_id']}.json"), "w") as f:
            json.dump({"job_id": st["pending_job_id"], "idx": st["cur_idx"],
                       "shots": st["pending_shots"], "counts": counts}, f)
        n = _fold(st, counts, value_fn)
        st["job_ids"].append({"idx": st["cur_idx"], "job_id": st["pending_job_id"], "shots": st["pending_shots"]})
        st["pending_job_id"], st["pending_shots"] = None, 0
        _save(spath, st)
        print(f"[recv] idx={st['cur_idx']} +{n} -> n_accum={st['n_accum']}")

        rad, mean = _radius(st)
        print(f"[check] n={st['n_accum']} mean={mean:+.4f} radius={rad:.5f} (eps={st['epsilon_stat']})")
        if rad < st["epsilon_stat"]:
            red = st["n_h"] / st["n_accum"]
            st["status"] = "stopped_ebs"
            st["result"] = {"estimate": mean, "n": st["n_accum"], "reduction": red}
            _save(spath, st)
            print(f"[STOP] EBS satisfied. estimate={mean:+.4f} realized n={st['n_accum']} ({red:.2f}x vs {st['n_h']})")
            print(f"       job_ids: {[j['job_id'] for j in st['job_ids']]}")
            return
        st["cur_idx"] += 1
        _save(spath, st)

    # ---- 2) cap? ----
    if st["cur_idx"] >= len(st["checkpoints"]):
        red = st["n_h"] / st["n_accum"]
        mean = st["sum_x"] / st["n_accum"]
        st["status"] = "stopped_cap"
        st["result"] = {"estimate": mean, "n": st["n_accum"], "reduction": red}
        _save(spath, st)
        print(f"[CAP] n={st['n_accum']} ({red:.2f}x)")
        return

    # ---- 3) submit next batch ----
    target = st["checkpoints"][st["cur_idx"]]
    delta_n = target - st["n_accum"]
    qc_t = transpile(qc, backend, optimization_level=1)
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    job = sampler.run([(qc_t,)], shots=delta_n)
    st["pending_job_id"], st["pending_shots"] = job.job_id(), delta_n
    _save(spath, st)
    print(f"[submit] idx={st['cur_idx']} -> checkpoint {target}: {delta_n} shots, job {job.job_id()} queued. Re-run later.")


if __name__ == "__main__":
    main()
