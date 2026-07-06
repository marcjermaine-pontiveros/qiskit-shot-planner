"""Reproduce the SWAP-test fidelity estimation (thesis PCSC anchor example).

Estimates the overlap F = |<ψ|φ>|² = E[Z_ancilla] of two single-qubit states
Ry(θ₁)|0>, Ry(θ₂)|0> (θ₁=0.3, θ₂=0.8) with three stopping rules. The ancilla
outcome map (qubit 0 → ±1) plus the exact ``statevector_sampler`` drive the
library planners; the dedicated ``run_swap_fidelity_estimator_ebs`` helper wraps
the same EBS path for the Aer/noisy case.

    PYTHONPATH=src python3 examples/pcsc2026/repro_swap.py
"""

from __future__ import annotations

import argparse
import csv
import os
from math import cos

import numpy as np

from qamp_shotplanner import (
    AnytimeEBSStopper,
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    statevector_sampler,
    swap_test_1qubit,
)

HERE = os.path.dirname(os.path.abspath(__file__))

THETA1 = 0.3
THETA2 = 0.8
EPS = 0.02
DELTA = 0.01
SEED = 42
N_TRIALS = 200


def ancilla_outcome(bitstring: str) -> float:
    """SWAP-test ancilla (qubit 0, rightmost bit): |0> → +1, |1> → -1."""
    return 1.0 if bitstring[-1] == "0" else -1.0


def _write_csv(name: str, rows: list[dict], fieldnames: list[str], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    circuit = swap_test_1qubit(THETA1, THETA2)  # unmeasured; ancilla = qubit 0
    f_ideal = cos((THETA1 - THETA2) / 2.0) ** 2
    sigma2 = 1.0 - f_ideal**2

    print(f"SWAP test: θ₁={THETA1}, θ₂={THETA2}, ε={EPS}, δ={DELTA}, seed={SEED}")
    print(f"  F_ideal={f_ideal:.4f}  σ²={sigma2:.4f}  trials={args.trials}")

    meta = np.random.default_rng(SEED)
    taus = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}
    covered = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}

    n_hoeff = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()

    for _ in range(args.trials):
        s = int(meta.integers(0, 2**31))

        samp = statevector_sampler(circuit, ancilla_outcome, seed=s)
        f_h = float(np.mean(samp(n_hoeff)))
        taus["Hoeffding"].append(n_hoeff)
        covered["Hoeffding"].append(abs(f_h - f_ideal) <= EPS)

        ebs = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
        r = ebs.run_batched(statevector_sampler(circuit, ancilla_outcome, seed=s + 1))
        taus["EBS-geom"].append(r.n)
        covered["EBS-geom"].append(abs(r.estimate - f_ideal) <= EPS)

        anytime = AnytimeEBSStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
        ra = anytime.run_batched(statevector_sampler(circuit, ancilla_outcome, seed=s + 2))
        taus["Anytime-EBS"].append(ra.n)
        covered["Anytime-EBS"].append(abs(ra.estimate - f_ideal) <= EPS)

    rows = []
    base = float(np.mean(taus["Hoeffding"]))
    for method in ("Hoeffding", "EBS-geom", "Anytime-EBS"):
        tau_mean = float(np.mean(taus[method]))
        cov = 100.0 * float(np.mean(covered[method]))
        reduction = base / tau_mean if tau_mean > 0 else float("nan")
        print(f"  {method:12s} τ={tau_mean:8.0f}  coverage={cov:5.1f}%  reduction={reduction:.2f}×")
        rows.append({
            "method": method,
            "theta1": THETA1,
            "theta2": THETA2,
            "eps": EPS,
            "delta": DELTA,
            "f_ideal": f"{f_ideal:.6f}",
            "sigma2": f"{sigma2:.6f}",
            "tau_mean": f"{tau_mean:.1f}",
            "coverage_pct": f"{cov:.1f}",
            "reduction": f"{reduction:.3f}",
        })

    path = _write_csv("swap.csv", rows, list(rows[0].keys()), args.out)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
