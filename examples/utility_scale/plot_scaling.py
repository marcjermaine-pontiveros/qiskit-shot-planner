"""Figure: the adaptive advantage grows with system size at utility scale.

For the TFIM magnetization M = (sum_i Z_i)/N at fixed evolution time, the
per-shot variance falls like ~1/N (averaging over N qubits), and since the
empirical Bernstein sample complexity scales with the variance, the shot
reduction over the fixed Hoeffding budget grows with N. This measures both.

    python3 examples/utility_scale/plot_scaling.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qamp_shotplanner import EmpiricalBernsteinStopper, HoeffdingPlanner
from tfim_magnetization import magnetization_sample_many, tfim_magnetization_circuit

EPS, DELTA, SEED, STEPS = 0.02, 0.01, 42, 10
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..")
N_LIST = [5, 10, 20, 40, 70]


def main():
    backend = AerSimulator(method="matrix_product_state")
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    stopper = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
    var, red = [], []
    for n_qubits in N_LIST:
        qc = tfim_magnetization_circuit(n_qubits, STEPS)
        qc_t = transpile(qc, basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=1)
        rng = np.random.default_rng(SEED + n_qubits)
        r = stopper.run_batched(magnetization_sample_many(qc_t, backend, rng))
        var.append(r.stats.variance_biased)
        red.append(n_h / r.n)
        print(f"N={n_qubits:3d}  sigma^2={var[-1]:.4f}  reduction={red[-1]:.2f}x")

    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    c1, c2 = "#1f77b4", "#d62728"
    ax1.plot(N_LIST, red, "-o", color=c1, lw=2, ms=7, label="EBS reduction (measured)")
    ax1.set_xlabel("system size $N$ (qubits)")
    ax1.set_ylabel(r"shot reduction vs. Hoeffding  ($n_H/\tau$)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xscale("log")
    ax1.set_xticks(N_LIST)
    ax1.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(N_LIST, var, "-s", color=c2, lw=2, ms=6, label=r"per-shot variance $\sigma^2$")
    # 1/N reference anchored at the first point
    ref = [var[0] * N_LIST[0] / n for n in N_LIST]
    ax2.plot(N_LIST, ref, "--", color=c2, alpha=0.5, label=r"$\propto 1/N$ reference")
    ax2.set_ylabel(r"per-shot variance $\sigma^2$", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="center right")
    ax1.set_title("Adaptive advantage grows with system size (TFIM $M$, $t=1.0$)")
    fig.tight_layout()

    out_dir = os.path.join(LIB, "results", "utility_scale")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"scaling.{ext}"), dpi=160)
    print("wrote", os.path.join(out_dir, "scaling.pdf"))


if __name__ == "__main__":
    main()
