"""Figure: the empirical Bernstein radius shrinks with shots until the tolerance.

Overlays two kinds of trajectory of the radius eps_n vs. accumulated shots n:
  (1) REAL hardware -- the live online SWAP-test run on ibm_fez, reconstructed
      from the 7 chained job records the stepper filed; and
  (2) SIMULATION -- the utility-scale TFIM magnetization observable at several
      evolution times (matrix-product-state backend).
Each curve descends across geometric checkpoints and stops the first time it
crosses the target eps. Low-variance observables (early time, or the averaged
magnetization) cross sooner -- the visual statement of why adaptive stopping
pays, and pays more at utility scale.

    python3 examples/utility_scale/plot_radius_shrink.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

from qamp_shotplanner import EmpiricalBernsteinStopper
from qamp_shotplanner.planners.empirical_bernstein import eb_radius_maurer
from qamp_shotplanner.stats.running_stats import RunningStats

from tfim_magnetization import magnetization_sample_many, tfim_magnetization_circuit

EPS, DELTA, SEED = 0.02, 0.01, 42
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..")


def _ancilla(bits):
    return 1.0 if bits.replace(" ", "")[-1] == "0" else -1.0


def hardware_swap_trajectory():
    """Reconstruct eps_n vs n from the filed job records of the live ibm_fez run."""
    sdir = os.path.join(LIB, "results", "online", "swap_maurer")
    st = json.load(open(os.path.join(sdir, "state.json")))
    R, alpha, deltas, cks = st["R"], st["alpha"], st["deltas"], st["checkpoints"]
    jobs = {}
    for f in glob.glob(os.path.join(sdir, "jobs", "*.json")):
        d = json.load(open(f))
        jobs[d["idx"]] = d
    n = nplus = 0
    ns, radii = [], []
    for idx in sorted(jobs):
        for bits, c in jobs[idx]["counts"].items():
            n += c
            if _ancilla(bits) > 0:
                nplus += c
        mean = (2 * nplus - n) / n
        var = max(0.0, 1.0 - mean * mean)
        r = eb_radius_maurer(n=n, R=R, var_biased=var, delta=deltas[idx])
        ns.append(n)
        radii.append(r)
    return ns, radii


def sim_trajectory(sample_many, stopper):
    """eps_n vs n across geometric checkpoints, stopping at first crossing."""
    stats, prev = RunningStats(), 0
    ns, radii = [], []
    for k, cp in enumerate(stopper.checkpoints()):
        dn = cp - prev
        if dn > 0:
            for x in sample_many(dn):
                stats.update(x)
            prev = cp
        if stats.n >= stopper.n_min:
            r = eb_radius_maurer(n=stats.n, R=stopper.R, var_biased=stats.variance_biased, delta=stopper._deltas[k])
            ns.append(stats.n)
            radii.append(r)
            if r < stopper.epsilon_stat:
                break
    return ns, radii


def _decorate(ax, eps_text_x):
    ax.axhspan(1e-3, EPS, color="green", alpha=0.06)
    ax.axhline(EPS, ls="--", color="black", lw=1.2)
    ax.text(eps_text_x, EPS * 1.15, r"target tolerance $\varepsilon = 0.02$", fontsize=9)
    ax.set_xlabel("accumulated shots $n$  (geometric checkpoints)")
    ax.set_ylabel(r"empirical Bernstein radius $\varepsilon_n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)


def main():
    backend = AerSimulator(method="matrix_product_state")
    stopper = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
    out_dir = os.path.join(LIB, "results", "utility_scale")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Figure 1: utility-scale TFIM magnetization only (simulation) ----
    from qiskit import transpile
    fig1, ax1 = plt.subplots(figsize=(7.0, 4.5))
    times = [(2, "t=0.2"), (10, "t=1.0"), (20, "t=2.0")]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.75, len(times)))
    for (steps, lbl), col in zip(times, cmap):
        qc_t = transpile(tfim_magnetization_circuit(20, steps),
                         basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=1)
        rng = np.random.default_rng(SEED + steps)
        ns, radii = sim_trajectory(magnetization_sample_many(qc_t, backend, rng), stopper)
        ax1.plot(ns, radii, "-o", ms=3, color=col, label=f"TFIM $M$, {lbl}")
        ax1.plot(ns[-1], radii[-1], "*", ms=13, color=col)
    _decorate(ax1, 1600)
    ax1.set_ylim(0.012, 3.0)
    ax1.set_title("Utility-scale magnetization: radius shrinks until tolerance ($N=20$)")
    ax1.legend(fontsize=8.5, loc="lower left", framealpha=0.95, title="evolution time (sim)")
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"radius_shrink_utility.{ext}"), dpi=160)

    # ---- Figure 2: live ibm_fez SWAP online run only (hardware) ----
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
    ns, radii = hardware_swap_trajectory()
    ax2.plot(ns, radii, "-s", ms=6, color="crimson", lw=2.4,
             label="SWAP fidelity (live ibm\\_fez, job-chained)")
    ax2.plot(ns[-1], radii[-1], "*", ms=18, color="crimson", zorder=6)
    ax2.annotate(f"stop @ {ns[-1]:,} shots\n({26492 / ns[-1]:.2f}$\\times$ realized)",
                 xy=(ns[-1], radii[-1]), xycoords="data",
                 xytext=(0.31, 0.14), textcoords="axes fraction", ha="left",
                 fontsize=9, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    _decorate(ax2, ns[0] * 1.1)
    ax2.set_ylim(0.015, 0.07)
    ax2.set_title("Realized online stopping on hardware (ibm\\_fez SWAP test)")
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(out_dir, f"radius_shrink_hardware.{ext}"), dpi=160)

    print("wrote radius_shrink_utility.pdf and radius_shrink_hardware.pdf")


if __name__ == "__main__":
    main()
