"""Reproduce shot savings vs. noise level (PCSC exp2 / QCE exp2, productized).

Sweeps a real Aer depolarizing noise model of increasing strength over both
workloads -- SWAP-test fidelity and QAOA MaxCut ⟨ZZ⟩ -- and reports EBS-geom /
Anytime-EBS stopping times against the fixed Hoeffding budget. As noise grows,
variance rises and the adaptive savings degrade toward the Hoeffding cap, the
paper's main empirical result. The SWAP path reproduces the PCSC anchor; the
QAOA path is the low-variance case.

The noise path is fully productized: ``noise_model_sampler`` +
``depolarizing_noise_model`` feed the library planners (no inline simulation).
This is the heaviest script (each trial pre-simulates a shot buffer); lower
``--trials`` for a quick look.

    PYTHONPATH=src python3 examples/pcsc2026/repro_noise_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from qamp_shotplanner import (
    AnytimeEBSStopper,
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    depolarizing_noise_model,
    noise_model_sampler,
    qaoa_maxcut_circuit,
    swap_test_1qubit,
    zz_outcome_map,
)

HERE = os.path.dirname(os.path.abspath(__file__))

EPS = 0.02
DELTA = 0.01
SEED = 42
N_TRIALS = 20

THETA1, THETA2 = 0.3, 0.8
GAMMA, BETA = 0.783, 0.438

# exp2/exp4 depolarizing parameterization: 1q rate p/10, 2q rate p.
NOISE_LEVELS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]


def _ancilla_outcome(bitstring: str) -> float:
    """SWAP-test ancilla (qubit 0, rightmost bit): |0> → +1, |1> → -1."""
    return 1.0 if bitstring[-1] == "0" else -1.0


def _workloads() -> list[tuple[str, object, object]]:
    return [
        ("swap", swap_test_1qubit(THETA1, THETA2), _ancilla_outcome),
        ("qaoa", qaoa_maxcut_circuit(GAMMA, BETA), zz_outcome_map),
    ]


def _write_csv(name: str, rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sweep(name: str, circuit, outcome_map, n_hoeff: int, trials: int) -> list[dict]:
    rows = []
    for p in NOISE_LEVELS:
        nm = depolarizing_noise_model(p1=p / 10.0, p2=p)
        ebs_taus, any_taus, sig2s = [], [], []
        for t in range(trials):
            ebs = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
            r = ebs.run_batched(noise_model_sampler(circuit, outcome_map, nm, seed=SEED + t))
            ebs_taus.append(r.n)
            sig2s.append(r.stats.variance_biased)

            anytime = AnytimeEBSStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
            ra = anytime.run_batched(
                noise_model_sampler(circuit, outcome_map, nm, seed=SEED + 1000 + t)
            )
            any_taus.append(ra.n)

        ebs_mean, any_mean = float(np.mean(ebs_taus)), float(np.mean(any_taus))
        sigma2 = float(np.mean(sig2s))
        red_ebs = n_hoeff / ebs_mean if ebs_mean > 0 else float("nan")
        red_any = n_hoeff / any_mean if any_mean > 0 else float("nan")
        print(f"  [{name}] p={p:.2f}  σ²≈{sigma2:.3f}  EBS={ebs_mean:8.0f} ({red_ebs:.2f}×)  "
              f"Anytime={any_mean:8.0f} ({red_any:.2f}×)")
        rows.append({
            "workload": name,
            "noise_p": f"{p:.2f}",
            "eps": EPS,
            "delta": DELTA,
            "sigma2_est": f"{sigma2:.6f}",
            "hoeff": n_hoeff,
            "ebs_mean": f"{ebs_mean:.1f}",
            "anytime_mean": f"{any_mean:.1f}",
            "reduction_ebs": f"{red_ebs:.3f}",
            "reduction_anytime": f"{red_any:.3f}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    n_hoeff = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    print(f"Noise sweep (SWAP + QAOA): ε={EPS}, δ={DELTA}, seed={SEED}, "
          f"trials={args.trials}, n_H={n_hoeff}")

    rows = []
    for name, circuit, outcome_map in _workloads():
        rows.extend(_sweep(name, circuit, outcome_map, n_hoeff, args.trials))

    path = _write_csv("noise_sweep.csv", rows, args.out)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
