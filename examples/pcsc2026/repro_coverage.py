"""Reproduce the empirical coverage validation (PCSC exp1).

Checks that all three stopping rules hold the prescribed (1-δ) coverage on a
depolarizing SWAP-test device model μ = (1-p)² (per-shot ±1, p₊ = (1+μ)/2),
swept over an (ε, δ, p) grid. The Bernoulli device stands in for the physical
channel; all EBS/Hoeffding math lives in the library planners.

    PYTHONPATH=src python3 examples/pcsc2026/repro_coverage.py
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
)

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 42
N_TRIALS = 500

# (epsilon, delta, noise_p)
PARAMS = [
    (0.05, 0.01, 0.00),
    (0.05, 0.01, 0.05),
    (0.02, 0.01, 0.00),
    (0.02, 0.01, 0.05),
]


def bernoulli_sampler(mu: float, seed: int):
    """Return sample_many(n) -> n i.i.d. ±1 draws with mean μ."""
    p_plus = (1.0 + mu) / 2.0
    rng = np.random.default_rng(seed)

    def sample_many(n: int) -> list[float]:
        return (2.0 * rng.binomial(1, p_plus, size=int(n)) - 1.0).tolist()

    return sample_many


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

    print(f"Coverage validation: seed={SEED}, trials={args.trials} "
          f"(target coverage ≥ 1-δ)")
    rows = []

    for eps, delta, p in PARAMS:
        mu = (1.0 - p) ** 2
        meta = np.random.default_rng(SEED)
        taus = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}
        covered = {"Hoeffding": [], "EBS-geom": [], "Anytime-EBS": []}

        n_hoeff = HoeffdingPlanner(epsilon_stat=eps, delta=delta, a=-1.0, b=1.0).planned_shots()

        for _ in range(args.trials):
            s = int(meta.integers(0, 2**31))

            mu_h = float(np.mean(bernoulli_sampler(mu, s)(n_hoeff)))
            taus["Hoeffding"].append(n_hoeff)
            covered["Hoeffding"].append(abs(mu_h - mu) <= eps)

            ebs = EmpiricalBernsteinStopper(epsilon_stat=eps, delta=delta, a=-1.0, b=1.0)
            r = ebs.run_batched(bernoulli_sampler(mu, s + 1))
            taus["EBS-geom"].append(r.n)
            covered["EBS-geom"].append(abs(r.estimate - mu) <= eps)

            anytime = AnytimeEBSStopper(epsilon_stat=eps, delta=delta, a=-1.0, b=1.0)
            ra = anytime.run_batched(bernoulli_sampler(mu, s + 2))
            taus["Anytime-EBS"].append(ra.n)
            covered["Anytime-EBS"].append(abs(ra.estimate - mu) <= eps)

        print(f"\n  ε={eps}  δ={delta}  p={p}  μ={mu:.4f}")
        for method in ("Hoeffding", "EBS-geom", "Anytime-EBS"):
            tau_mean = float(np.mean(taus[method]))
            tau_std = float(np.std(taus[method]))
            cov = 100.0 * float(np.mean(covered[method]))
            print(f"    {method:12s} coverage={cov:5.1f}%  τ={tau_mean:8.0f} ± {tau_std:.0f}")
            rows.append({
                "eps": eps,
                "delta": delta,
                "noise_p": p,
                "mu": f"{mu:.6f}",
                "method": method,
                "coverage_pct": f"{cov:.1f}",
                "tau_mean": f"{tau_mean:.1f}",
                "tau_std": f"{tau_std:.1f}",
            })

    path = _write_csv("coverage.csv", rows, list(rows[0].keys()), args.out)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
