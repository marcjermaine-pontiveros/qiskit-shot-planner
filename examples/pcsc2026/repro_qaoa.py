"""Reproduce the QAOA MaxCut energy-estimation result (PCSC exp5).

Estimates <Z⊗Z> of the 2-qubit p=1 MaxCut QAOA state (γ=0.783, β=0.438, a
low-variance operating point) with three stopping rules and reports mean
stopping time, empirical coverage, and shot reduction vs. Hoeffding.

Circuit, observable, and ideal reference come from ``workloads.qaoa``; the
exact-distribution ``statevector_sampler`` feeds the library planners. No
EBS/QAOA math is inlined here.

    PYTHONPATH=src python3 examples/pcsc2026/repro_qaoa.py
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
    ideal_zz,
    qaoa_maxcut_circuit,
    statevector_sampler,
    zz_outcome_map,
)

HERE = os.path.dirname(os.path.abspath(__file__))

GAMMA = 0.783
BETA = 0.438
EPS = 0.02
DELTA = 0.01
SEED = 42
N_TRIALS = 200


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

    circuit = qaoa_maxcut_circuit(GAMMA, BETA)
    mu_ideal = ideal_zz(GAMMA, BETA)
    sigma2 = 1.0 - mu_ideal**2

    print(f"QAOA MaxCut: γ={GAMMA}, β={BETA}, ε={EPS}, δ={DELTA}, seed={SEED}")
    print(f"  <ZZ>_ideal={mu_ideal:.4f}  σ²={sigma2:.4f}  trials={args.trials}")

    meta = np.random.default_rng(SEED)
    taus = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}
    covered = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}

    hoeff = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
    n_hoeff = hoeff.planned_shots()

    for _ in range(args.trials):
        s = int(meta.integers(0, 2**31))

        samp = statevector_sampler(circuit, zz_outcome_map, seed=s)
        mu_h = float(np.mean(samp(n_hoeff)))
        taus["Hoeffding"].append(n_hoeff)
        covered["Hoeffding"].append(abs(mu_h - mu_ideal) <= EPS)

        ebs = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
        r = ebs.run_batched(statevector_sampler(circuit, zz_outcome_map, seed=s + 1))
        taus["EBS-geom"].append(r.n)
        covered["EBS-geom"].append(abs(r.estimate - mu_ideal) <= EPS)

        anytime = AnytimeEBSStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
        ra = anytime.run_batched(statevector_sampler(circuit, zz_outcome_map, seed=s + 2))
        taus["Anytime-EBS"].append(ra.n)
        covered["Anytime-EBS"].append(abs(ra.estimate - mu_ideal) <= EPS)

    rows = []
    base = float(np.mean(taus["Hoeffding"]))
    for method in ("Hoeffding", "EBS-geom", "Anytime-EBS"):
        tau_mean = float(np.mean(taus[method]))
        cov = 100.0 * float(np.mean(covered[method]))
        reduction = base / tau_mean if tau_mean > 0 else float("nan")
        print(f"  {method:12s} τ={tau_mean:8.0f}  coverage={cov:5.1f}%  reduction={reduction:.2f}×")
        rows.append({
            "method": method,
            "gamma": GAMMA,
            "beta": BETA,
            "eps": EPS,
            "delta": DELTA,
            "mu_ideal": f"{mu_ideal:.6f}",
            "sigma2": f"{sigma2:.6f}",
            "tau_mean": f"{tau_mean:.1f}",
            "coverage_pct": f"{cov:.1f}",
            "reduction": f"{reduction:.3f}",
        })

    path = _write_csv("qaoa.csv", rows, list(rows[0].keys()), args.out)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
