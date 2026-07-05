"""Reproduce the multi-observable H2 VQE energy estimate (QCE exp_qce3).

Estimates the H2 ground-state energy E = Σ_j c_j <P_j> over a bond-length sweep
by running an Empirical Bernstein stopper independently per Pauli term via
``bonferroni_estimate`` (δ split δ/m, accuracy split ε/(2‖c‖₁)). Reports total
shots vs. the per-term Bonferroni Hoeffding baseline and the energy error.

Ansatz, coefficients, angles, and per-term outcome maps come from
``workloads.vqe_h2``; exact energies use ``SparsePauliOp`` for reference only.

    PYTHONPATH=src python3 examples/pcsc2026/repro_vqe_h2.py
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from qamp_shotplanner import (
    H2_ANGLES,
    HoeffdingPlanner,
    bonferroni_estimate,
    h2_terms,
    pauli_outcome_map,
    statevector_sampler,
    vqe_ansatz,
)

HERE = os.path.dirname(os.path.abspath(__file__))

EPS = 0.02
DELTA = 0.01
SEED = 42
N_TRIALS = 100
N_TERMS = 4  # II, IZ, ZI, ZZ


def _write_csv(name: str, rows: list[dict], fieldnames: list[str], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def exact_energy(circuit, terms) -> float:
    sv = Statevector.from_instruction(circuit)
    return float(sum(c * sv.expectation_value(SparsePauliOp(label)).real for c, label in terms))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    print(f"H2 VQE (STO-3G): ε={EPS}, δ={DELTA}, seed={SEED}, trials={args.trials}")

    bond_lengths = sorted(H2_ANGLES)
    meta = np.random.default_rng(SEED)
    rows = []

    for r in bond_lengths:
        t0, t1 = H2_ANGLES[r]
        circuit = vqe_ansatz(t0, t1)
        terms = h2_terms(r)
        e_exact = exact_energy(circuit, terms)

        # Matched Hoeffding baseline: same per-term budget bonferroni_estimate uses
        # (δ_j = δ/m, ε_j = ε/(2‖c‖₁)), so the reduction is a like-for-like ratio.
        l1 = sum(abs(c) for c, _ in terms)
        eps_j = EPS / (2.0 * l1)
        n_hoeff_total = N_TERMS * HoeffdingPlanner(
            epsilon_stat=eps_j, delta=DELTA / N_TERMS, a=-1.0, b=1.0
        ).planned_shots()

        totals, energies = [], []
        for _ in range(args.trials):
            samplers = [
                (c, statevector_sampler(circuit, pauli_outcome_map(label),
                                        seed=int(meta.integers(0, 2**31))))
                for c, label in terms
            ]
            res = bonferroni_estimate(samplers, eps=EPS, delta=DELTA)
            totals.append(res.total_shots)
            energies.append(res.energy)

        ebs_mean = float(np.mean(totals))
        e_mean = float(np.mean(energies))
        ratio = n_hoeff_total / ebs_mean if ebs_mean > 0 else float("nan")
        e_err = abs(e_mean - e_exact)
        print(f"  r={r:.2f}Å  E_exact={e_exact:+.4f}  E_EBS={e_mean:+.4f}  "
              f"|ΔE|={e_err:.4f}  EBS_total={ebs_mean:.0f}  reduction={ratio:.2f}×")
        rows.append({
            "bond_length": f"{r:.2f}",
            "eps": EPS,
            "delta": DELTA,
            "e_exact": f"{e_exact:.6f}",
            "e_ebs_mean": f"{e_mean:.6f}",
            "energy_abs_err": f"{e_err:.6f}",
            "hoeff_total": n_hoeff_total,
            "ebs_total_mean": f"{ebs_mean:.1f}",
            "reduction": f"{ratio:.3f}",
        })

    path = _write_csv("vqe_h2.csv", rows, list(rows[0].keys()), args.out)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
