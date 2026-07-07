"""Larger QAOA MaxCut: adaptive stopping vs graph size (6, 10, 14 qubits).

Answers the QCE reviewer concern that every reported experiment is small
(<= 3 qubits) by scaling the QAOA MaxCut cost estimation up to 14 qubits in
exact statevector simulation, and characterizing how the shot reduction behaves
with system size.

Setup
-----
For a graph G = (V, E) the p=1 QAOA MaxCut cost Hamiltonian is
``C = sum_{(i,j) in E} Z_i Z_j`` (up to the usual affine (1 - ZZ)/2 rescaling).
Every ``Z_i Z_j`` is diagonal in the computational basis, so ALL edges commute
and are read from a SINGLE Z-basis measurement circuit -- exactly the shared-shot
grouping the thesis uses for commuting families. We estimate the INTENSIVE cost
density

    e = C / |E| = (1/|E|) sum_{(i,j) in E} <Z_i Z_j>      in [-1, 1],

a single bounded observable, with the Empirical Bernstein stopper and compare
its stopping time against the fixed Hoeffding budget for the same (eps, delta).

The graph is a deterministic circulant ring-plus-chords: edges (i, i+1) and
(i, i+2) modulo N (4-regular, |E| = 2N). The p=1 angles gamma=0.5, beta=0.2 are
FIXED across all sizes and give a non-trivial, low-variance density (the density
concentrates as more edges are averaged, so the adaptive advantage GROWS with N).

Everything runs on the exact statevector (N <= 16). Qiskit 2 only.

    python examples/larger_scale/qaoa_scaling.py
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    statevector_value_sampler,
)

SIZES = (6, 10, 14)
GAMMA, BETA = 0.5, 0.2         # fixed p=1 angles across all sizes
EPS, DELTA = 0.02, 0.01
SEEDS = (42, 43, 44, 45, 46)   # master seeds; per-seed mean, then mean +/- std
TRIALS = 20                    # trials per seed


def circulant_graph(n: int) -> list[tuple[int, int]]:
    """Ring-plus-chords: edges (i, i+1) and (i, i+2) mod n (4-regular, |E| = 2n)."""
    edges = set()
    for i in range(n):
        edges.add(tuple(sorted((i, (i + 1) % n))))
        edges.add(tuple(sorted((i, (i + 2) % n))))
    return sorted(edges)


def qaoa_maxcut_circuit(n: int, edges: list[tuple[int, int]],
                        gamma: float, beta: float) -> QuantumCircuit:
    """Build the p=1 MaxCut QAOA state for ``edges`` (no measurement).

    Phase separation exp(-i gamma Z_i Z_j) is CX(i,j) . Rz(2 gamma, j) . CX(i,j);
    the mixer exp(-i beta sum_q X_q) is Rx(2 beta) on every qubit.
    """
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i, j in edges:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    for q in range(n):
        qc.rx(2 * beta, q)
    return qc


def cost_density_value_map(edges: list[tuple[int, int]]):
    """Per-shot intensive cost density: (sum over edges of Z_i Z_j parity) / |E|."""
    m = len(edges)

    def value(bitstring: str) -> float:
        bits = bitstring.replace(" ", "")
        n = len(bits)
        total = 0
        for i, j in edges:
            bi = int(bits[n - 1 - i])   # big-endian label; rightmost char is qubit 0
            bj = int(bits[n - 1 - j])
            total += 1 - 2 * (bi ^ bj)  # +1 if aligned, -1 if cut
        return total / m

    return value


def exact_density(circuit: QuantumCircuit, value) -> float:
    """Exact mean of the density observable = sum_i P(i) * value(i)."""
    n = circuit.num_qubits
    probs = Statevector.from_instruction(circuit).probabilities()
    return float(sum(p * value(format(i, f"0{n}b")) for i, p in enumerate(probs)))


def run_size(n: int, n_hoeffding: int) -> dict:
    """EBS stopping for the cost density at size ``n``, over SEEDS x TRIALS."""
    edges = circulant_graph(n)
    circuit = qaoa_maxcut_circuit(n, edges, GAMMA, BETA)
    value = cost_density_value_map(edges)
    e_exact = exact_density(circuit, value)
    variance = float(
        sum(p * (value(format(i, f"0{n}b")) - e_exact) ** 2
            for i, p in enumerate(Statevector.from_instruction(circuit).probabilities()))
    )

    per_seed_tau, per_seed_cov = [], []
    for master in SEEDS:
        mrng = np.random.default_rng(master)
        taus, covs = [], []
        for _ in range(TRIALS):
            sd = int(mrng.integers(0, 2**31))
            sampler = statevector_value_sampler(circuit, value, seed=sd)
            r = EmpiricalBernsteinStopper(
                epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
            ).run_batched(sampler)
            taus.append(r.n)
            covs.append(abs(r.estimate - e_exact) <= EPS)
        per_seed_tau.append(float(np.mean(taus)))
        per_seed_cov.append(100.0 * float(np.mean(covs)))

    tau_mean = float(np.mean(per_seed_tau))
    tau_std = float(np.std(per_seed_tau))
    cov_mean = float(np.mean(per_seed_cov))
    return {
        "n": n,
        "edges": len(edges),
        "e_exact": e_exact,
        "variance": variance,
        "n_hoeffding": n_hoeffding,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "reduction": n_hoeffding / tau_mean,
        "coverage": cov_mean,
    }


def main() -> None:
    n_h = HoeffdingPlanner(
        epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
    ).planned_shots()

    print("Larger QAOA MaxCut scaling  (p=1, gamma=0.5, beta=0.2, "
          f"eps={EPS}, delta={DELTA})")
    print(f"Cost density e = C/|E| in [-1, 1];  fixed Hoeffding budget = {n_h:,} shots")
    print(f"Seeds = {list(SEEDS)}, {TRIALS} trials each  (statevector, exact)\n")

    header = (f"{'qubits':>7} {'edges':>6} {'e_exact':>9} {'var':>7} "
              f"{'EBS shots':>16} {'reduction':>10} {'coverage':>9}")
    print(header)
    print("-" * len(header))
    rows = []
    for n in SIZES:
        row = run_size(n, n_h)
        rows.append(row)
        print(f"{row['n']:>7} {row['edges']:>6} {row['e_exact']:>+9.4f} "
              f"{row['variance']:>7.4f} "
              f"{row['tau_mean']:>8.0f} +/- {row['tau_std']:<4.0f} "
              f"{row['reduction']:>9.2f}x {row['coverage']:>8.1f}%")

    print("\nTakeaway: the density variance falls as the graph grows "
          "(more edges averaged),")
    print("so the adaptive reduction over the fixed Hoeffding budget GROWS with size,")
    print("while coverage against the exact statevector cost stays at ~100%.")


if __name__ == "__main__":
    main()
