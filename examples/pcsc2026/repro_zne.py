"""Reproduce ZNE + adaptive stopping complementarity (PCSC exp4).

Error mitigation reduces hardware bias; adaptive stopping controls sampling
error — orthogonal knobs that stack. On a Bell state under depolarizing noise
(⟨ZZ⟩_ideal = 1) the noise is amplified by global gate folding (factors 1..5),
⟨ZZ⟩ is Richardson-extrapolated to zero noise, and per-factor Hoeffding vs.
EBS-geom shot costs are reported.

Folding and extrapolation come from ``mitigation.zne``; the noisy sampler and
planners from the library. No inline ZNE/EBS math.

    PYTHONPATH=src python3 examples/pcsc2026/repro_zne.py
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from qiskit import QuantumCircuit

from qamp_shotplanner import (
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    depolarizing_noise_model,
    fold_gates,
    noise_model_sampler,
    zne_extrapolate,
    zz_outcome_map,
)

HERE = os.path.dirname(os.path.abspath(__file__))

NOISE_P = 0.05
EPS = 0.02
DELTA = 0.01
SEED = 42
N_SHOTS = 8_000   # shots per noise-factor ⟨ZZ⟩ estimate
N_TRIALS = 50     # EBS trials per noise factor
NOISE_FACTORS = [1, 2, 3, 4, 5]
IDEAL_ZZ = 1.0


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


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
    ap.add_argument("--shots", type=int, default=N_SHOTS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    nm = depolarizing_noise_model(p1=NOISE_P / 10.0, p2=NOISE_P)
    base = bell_circuit()
    n_hoeff = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()

    print(f"ZNE + adaptive stopping: p={NOISE_P}, ε={EPS}, δ={DELTA}, seed={SEED}")
    print(f"  factors={NOISE_FACTORS}  shots/estimate={args.shots}  trials={args.trials}")

    zz_values, rows = [], []
    for f in NOISE_FACTORS:
        folded = fold_gates(base, f)
        sampler = noise_model_sampler(folded, zz_outcome_map, nm, seed=SEED, buffer_size=args.shots)
        zz = float(np.mean(sampler(args.shots)))
        zz_values.append(zz)
        bias = abs(zz - IDEAL_ZZ)

        ebs_taus = []
        for t in range(args.trials):
            ebs = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
            r = ebs.run_batched(
                noise_model_sampler(folded, zz_outcome_map, nm, seed=SEED + t)
            )
            ebs_taus.append(r.n)
        ebs_mean = float(np.mean(ebs_taus))
        ratio = n_hoeff / ebs_mean if ebs_mean > 0 else float("nan")
        print(f"  f={f}: ⟨ZZ⟩={zz:.4f}  bias={bias:.4f}  "
              f"Hoeff={n_hoeff}  EBS={ebs_mean:.0f}  reduction={ratio:.2f}×")
        rows.append({
            "kind": "raw",
            "label": f"f={f}",
            "zz": f"{zz:.6f}",
            "bias": f"{bias:.6f}",
            "tau_hoeff": n_hoeff,
            "tau_ebs_mean": f"{ebs_mean:.1f}",
            "reduction": f"{ratio:.3f}",
        })

    zz_lin = zne_extrapolate(NOISE_FACTORS, zz_values, method="linear")
    zz_exp = zne_extrapolate(NOISE_FACTORS, zz_values, method="exponential")
    print(f"\n  raw (f=1) ⟨ZZ⟩={zz_values[0]:.4f}  bias={abs(zz_values[0] - IDEAL_ZZ):.4f}")
    print(f"  ZNE-linear ⟨ZZ⟩={zz_lin:.4f}  bias={abs(zz_lin - IDEAL_ZZ):.4f}")
    print(f"  ZNE-exp    ⟨ZZ⟩={zz_exp:.4f}  bias={abs(zz_exp - IDEAL_ZZ):.4f}")
    for label, val in [("ZNE-linear", zz_lin), ("ZNE-exp", zz_exp)]:
        rows.append({
            "kind": "zne",
            "label": label,
            "zz": f"{val:.6f}",
            "bias": f"{abs(val - IDEAL_ZZ):.6f}",
            "tau_hoeff": n_hoeff,
            "tau_ebs_mean": rows[0]["tau_ebs_mean"],
            "reduction": rows[0]["reduction"],
        })

    path = _write_csv("zne.csv", rows, list(rows[0].keys()), args.out)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
