"""The geometric ratio beta as an overshoot vs. checkpoint-count trade-off.

beta is a design knob, not a proven optimum. Smaller beta places checkpoints
more densely, so the rule stops closer to the ideal stopping point n* (less
overshoot, fewer realized shots) but at more checkpoints K = log_beta(n_H/n_min)
(each spending failure budget delta/K). This sweep measures both, showing why
beta ~ 1.1 is the simulation operating point while the hardware runs used
beta = 1.35 (fewer job submissions K).

    python3 scripts/beta_sweep.py
"""

from __future__ import annotations

import os
from math import sqrt

import numpy as np

from qamp_shotplanner import EmpiricalBernsteinStopper, HoeffdingPlanner

EPS, DELTA = 0.02, 0.01
SIGMA2, SEEDS, TRIALS = 0.05, [42, 43, 44, 45, 46], 400
BETAS = [1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0]
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..")


def bernoulli(mu, rng):
    p = (1.0 + mu) / 2.0

    def sample_many(n):
        return (2.0 * (rng.random(int(n)) < p) - 1.0).tolist()

    return sample_many


def main():
    mu = sqrt(1.0 - SIGMA2)
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    print(f"beta sweep at sigma^2={SIGMA2} (mu={mu:.3f}), eps={EPS}, delta={DELTA}, n_H={n_h}")
    print(f"{'beta':>6} {'checkpoints K':>14} {'mean tau':>10} {'overshoot vs beta=1.05':>24}")
    rows = []
    base = None
    for beta in BETAS:
        stp = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0, beta=beta)
        k = len(stp.checkpoints())
        taus = []
        for master in SEEDS:
            mrng = np.random.default_rng(master)
            for _ in range(TRIALS):
                taus.append(EmpiricalBernsteinStopper(
                    epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0, beta=beta
                ).run_batched(bernoulli(mu, np.random.default_rng(int(mrng.integers(0, 2**31))))).n)
        mt = float(np.mean(taus))
        base = base or mt
        print(f"{beta:6.2f} {k:14d} {mt:10.0f} {mt/base:23.3f}x")
        rows.append((beta, k, mt))
    _plot(rows, n_h)
    return rows


def _plot(rows, n_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = [r[0] for r in rows]
    ks = [r[1] for r in rows]
    taus = [r[2] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    ax1.plot(betas, taus, "-o", color="#d62728", label="mean realized shots $\\mathbb{E}[\\tau]$ (overshoot)")
    ax1.set_xlabel(r"geometric ratio $\beta$")
    ax1.set_ylabel(r"mean realized shots $\mathbb{E}[\tau]$", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.axvline(1.1, ls=":", color="gray"); ax1.axvline(1.35, ls=":", color="gray")
    ax1.text(1.11, min(taus), "sim", fontsize=8, color="gray")
    ax1.text(1.36, min(taus), "hw", fontsize=8, color="gray")
    ax2 = ax1.twinx()
    ax2.plot(betas, ks, "-s", color="#1f77b4", label="checkpoints $K$")
    ax2.set_ylabel(r"checkpoint count $K$", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_yscale("log")
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="upper center")
    ax1.set_title(r"$\beta$ trades overshoot against checkpoint count")
    ax1.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(LIB, "results", "beta_sweep")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"beta_sweep.{ext}"), dpi=160)
    print("wrote", os.path.join(out, "beta_sweep.pdf"))


if __name__ == "__main__":
    main()
