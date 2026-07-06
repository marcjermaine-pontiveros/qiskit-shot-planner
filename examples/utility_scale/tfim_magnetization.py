"""Adaptive shot planning for a utility-scale observable: TFIM magnetization.

Faithfully reproduces the observable of IBM's utility-scale course notebook
``utility-ii`` -- the normalized magnetization M = (sum_i Z_i)/N of a 1-D
transverse-field Ising model under Lie-Trotter time evolution (J=1, h=-5,
dt=0.1) -- and plans its shot budget with the empirical Bernstein rule instead
of the notebook's fixed default precision.

The point the sweep makes: M is an AVERAGE over N qubits, so its per-shot
variance is suppressed (sigma^2 <= (1 - mean^2), and averaging pulls it well
below the single-Pauli worst case). The empirical Bernstein advantage scales
like 1/sigma^2, so adaptive stopping does not merely survive at utility scale --
it pays MORE there. This script measures that empirically; it does not assume it.

    python3 examples/utility_scale/tfim_magnetization.py                 # N=20 time sweep (Aer)
    python3 examples/utility_scale/tfim_magnetization.py --scan-n        # variance-vs-N study
"""

from __future__ import annotations

import argparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator

from qamp_shotplanner import EmpiricalBernsteinStopper, HoeffdingPlanner

EPS, DELTA, SEED = 0.02, 0.01, 42
J, H, DT = 1.0, -5.0, 0.1


def tfim_hamiltonian(n_qubits: int) -> SparsePauliOp:
    zz = [("ZZ", [i, i + 1], -J) for i in range(n_qubits - 1)]
    x = [("X", [i], -H) for i in range(n_qubits)]
    return SparsePauliOp.from_sparse_list([*zz, *x], num_qubits=n_qubits).simplify()


def tfim_magnetization_circuit(n_qubits: int, n_steps: int) -> QuantumCircuit:
    """|0...0> evolved by n_steps Lie-Trotter steps, all qubits measured in Z."""
    step = PauliEvolutionGate(tfim_hamiltonian(n_qubits), DT, synthesis=LieTrotter())
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        qc.append(step, qc.qubits)
    qc.measure_all()
    return qc


def magnetization_sample_many(circuit, backend, rng):
    """Return a sample_many(n): each shot's per-shot M = (sum_i z_i)/N in [-1,1]."""
    n_qubits = circuit.num_qubits
    tqc = backend.run  # bind
    compiled = circuit

    def sample_many(n):
        seed = int(rng.integers(0, 2**31))
        counts = tqc(compiled, shots=int(n), seed_simulator=seed).result().get_counts()
        samples = []
        for bits, c in counts.items():
            b = bits.replace(" ", "")
            # z_i = +1 for bit '0', -1 for bit '1'; M = mean over qubits
            ones = b.count("1")
            m = (n_qubits - 2 * ones) / n_qubits
            samples.extend([m] * c)
        return samples

    return sample_many


def _plan(circuit, backend, seed):
    rng = np.random.default_rng(seed)
    sm = magnetization_sample_many(circuit, backend, rng)
    stopper = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
    return stopper.run_batched(sm)


def time_sweep(n_qubits, n_steps_max, backend):
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    print(f"TFIM magnetization time sweep: N={n_qubits}, eps={EPS}, delta={DELTA}, n_H={n_h}")
    print(f"{'t':>5} {'step':>5} {'M_hat':>8} {'sigma^2':>9} {'EBS tau':>9} {'reduction':>10} {'stop':>6}")
    for k in range(n_steps_max + 1):
        qc = tfim_magnetization_circuit(n_qubits, k)
        qc_t = _transpile(qc, backend)
        r = _plan(qc_t, backend, SEED + k)
        var = r.stats.variance_biased
        print(f"{k*DT:5.1f} {k:5d} {r.estimate:+8.4f} {var:9.4f} {r.n:9d} "
              f"{n_h / r.n:9.2f}x {r.stopped_by:>6}")


def scan_n(n_list, n_steps, backend):
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    print(f"Variance-vs-N study at t={n_steps*DT:.1f} ({n_steps} steps), eps={EPS}, n_H={n_h}")
    print(f"{'N':>5} {'M_hat':>8} {'sigma^2':>9} {'EBS tau':>9} {'reduction':>10}")
    for n_qubits in n_list:
        qc = tfim_magnetization_circuit(n_qubits, n_steps)
        qc_t = _transpile(qc, backend)
        r = _plan(qc_t, backend, SEED + n_qubits)
        print(f"{n_qubits:5d} {r.estimate:+8.4f} {r.stats.variance_biased:9.4f} "
              f"{r.n:9d} {n_h / r.n:9.2f}x")


def _transpile(qc, backend):
    # decompose to basis gates WITHOUT a coupling map: the MPS simulator has
    # all-to-all connectivity, and binding to a device target caps N at 63.
    from qiskit import transpile
    return transpile(qc, basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-qubits", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--scan-n", action="store_true", help="variance-vs-N study at fixed time")
    args = ap.parse_args()
    # matrix_product_state scales to N=70 (short-time TFIM is low-entanglement);
    # statevector would need 2^N amplitudes and OOMs past ~30 qubits.
    backend = AerSimulator(method="matrix_product_state")
    if args.scan_n:
        scan_n([5, 10, 20, 40, 70], n_steps=args.steps, backend=backend)
    else:
        time_sweep(args.n_qubits, args.steps, backend)


if __name__ == "__main__":
    main()
