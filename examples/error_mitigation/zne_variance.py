"""ZNE variance-inflation and the mitigation-worthwhile diagnostic.

Error mitigation trades bias for variance: zero-noise extrapolation (ZNE) forms
an (near-)unbiased estimate of the ideal value as a linear combination of
noise-scaled measurements, which inflates the per-shot variance by a factor
gamma^2 = sum_i w_i^2 and widens the estimator's range. The empirical Bernstein
rule *measures* that inflation directly from the shot stream, so the overhead
gamma^2 ~ tau_mit / tau_unmit and the removed bias b ~ |mu_zne - mu_dev| both
fall out as certified quantities. Sweeping a single noise knob p from low to
high shows the three regimes: mitigation not worth it (bias < eps), worth it
(bias > eps, overhead affordable), and not enough (variance blows the budget).

Noise model (per the thesis calibrated model), scaled by the knob p:
  - single-qubit depolarizing rate p on {h, rx, ry, rz, x, sx}
  - two-qubit depolarizing rate min(5p, 1) on {cx, cz}   (2q gates ~5x noisier)
  - symmetric readout bit-flip rate min(p, 0.5)
ZNE scales noise by lambda in {1,2,3} (running at rate lambda*p) and extrapolates
to lambda=0 by a linear least-squares fit: mu_zne = (4 y1 + y2 - 2 y3)/3.

    python3 examples/error_mitigation/zne_variance.py
"""

from __future__ import annotations

import os
from math import cos

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from qamp_shotplanner import EmpiricalBernsteinStopper, HoeffdingPlanner, swap_test_1qubit

EPS, DELTA, SEED = 0.02, 0.01, 42
THETA1, THETA2 = 0.3, 0.8
F_IDEAL = cos((THETA1 - THETA2) / 2.0) ** 2          # SWAP-test fidelity = <Z_anc>
LAMBDAS = [1, 2, 3]
ZNE_W = np.array([4.0, 1.0, -2.0]) / 3.0             # linear LSQ extrapolation to lambda=0
R_ZNE = 2.0 * float(np.sum(np.abs(ZNE_W)))           # range of the combined estimator (=14/3)
GAMMA2_NOMINAL = float(np.sum(ZNE_W ** 2))           # nominal variance inflation (=7/3)
NOISE_GRID = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "..")


def noise_model(p: float) -> NoiseModel:
    nm = NoiseModel()
    p1, p2, pr = min(p, 1.0), min(5 * p, 1.0), min(p, 0.5)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "rx", "ry", "rz", "x", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - pr, pr], [pr, 1 - pr]]))
    return nm


def _ancilla(bits: str) -> float:
    return 1.0 if bits.replace(" ", "")[-1] == "0" else -1.0


def _shots_pm1(circuit, p_scale, n, rng):
    """n per-shot +/-1 ancilla outcomes under the noise model at rate p_scale."""
    backend = AerSimulator(noise_model=noise_model(p_scale))
    qc = transpile(circuit, backend, optimization_level=1)
    counts = backend.run(qc, shots=int(n), seed_simulator=int(rng.integers(0, 2**31))).result().get_counts()
    out = []
    for bits, c in counts.items():
        out.extend([_ancilla(bits)] * c)
    rng.shuffle(out)
    return np.asarray(out, dtype=float)


def unmitigated_sampler(circuit, p, rng):
    def sample_many(n):
        return _shots_pm1(circuit, p, n, rng).tolist()
    return sample_many


def zne_sampler(circuit, p, rng):
    """Each round combines one shot at each lambda into a single ZNE sample."""
    def sample_many(n):
        cols = [_shots_pm1(circuit, lam * p, n, rng) for lam in LAMBDAS]
        return (ZNE_W[0] * cols[0] + ZNE_W[1] * cols[1] + ZNE_W[2] * cols[2]).tolist()
    return sample_many


def _run(stopper, sample_many):
    return stopper.run_batched(sample_many)


def main():
    qc = swap_test_1qubit(THETA1, THETA2)
    qc.measure_all()
    n_h_raw = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    stop_raw = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0)
    stop_zne = EmpiricalBernsteinStopper(epsilon_stat=EPS, delta=DELTA, a=-R_ZNE / 2, b=R_ZNE / 2)

    print(f"ZNE diagnostic on SWAP test (F_ideal={F_IDEAL:.4f}), eps={EPS}, delta={DELTA}")
    print(f"ZNE lambda={LAMBDAS}, weights={ZNE_W.round(3).tolist()}, nominal gamma^2={GAMMA2_NOMINAL:.2f}, "
          f"range R_zne={R_ZNE:.2f}  (unmitigated n_H={n_h_raw})")
    print(f"{'p':>8} {'mu_dev':>8} {'bias_dev':>9} {'mu_zne':>8} {'bias_zne':>9} "
          f"{'tau_un':>8} {'tau_zne':>9} {'gamma^2':>8} {'verdict':>16}")
    rows = []
    for p in NOISE_GRID:
        ru = _run(stop_raw, unmitigated_sampler(qc, p, np.random.default_rng(SEED)))
        rz = _run(stop_zne, zne_sampler(qc, p, np.random.default_rng(SEED + 1)))
        tau_un = ru.n
        tau_zne = 3 * rz.n                       # 3 circuit executions per ZNE round
        bias_dev = abs(ru.estimate - F_IDEAL)
        bias_zne = abs(rz.estimate - F_IDEAL)
        gamma2 = tau_zne / tau_un
        if bias_dev <= EPS:
            verdict = "not worth it"           # bias already within tolerance; tax buys nothing
        elif bias_zne <= EPS:
            verdict = "worth it"               # tax buys a within-tolerance answer
        else:
            verdict = "not enough"             # tax paid, residual bias still > eps
        print(f"{p:8.4f} {ru.estimate:8.4f} {bias_dev:9.4f} {rz.estimate:8.4f} {bias_zne:9.4f} "
              f"{tau_un:8d} {tau_zne:9d} {gamma2:8.2f} {verdict:>16}")
        rows.append((p, bias_dev, bias_zne, tau_un, tau_zne, gamma2, verdict))
    _plot(rows)
    return rows


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = [r[0] for r in rows]
    bd = [r[1] for r in rows]
    bz = [r[2] for r in rows]
    g2 = [r[5] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(ps, bd, "-o", color="#d62728", label="bias, unmitigated $|\\hat\\mu_{dev}-\\mu_{ideal}|$")
    ax1.plot(ps, bz, "-s", color="#2ca02c", label="bias, ZNE $|\\hat\\mu_{zne}-\\mu_{ideal}|$")
    ax1.axhline(EPS, ls="--", color="black", lw=1)
    ax1.text(ps[0], EPS * 1.25, r"tolerance $\varepsilon=0.02$", fontsize=8.5)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("noise knob $p$ (single-qubit depolarizing rate)")
    ax1.set_ylabel("bias to ideal")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(ps, g2, "-^", color="#1f77b4", label=r"measured overhead $\gamma^2=\tau_{mit}/\tau_{unmit}$")
    ax2.set_ylabel(r"measured overhead $\gamma^2$", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    lines = ax1.get_lines()[:2] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center left")
    ax1.set_title("ZNE trades bias for a measured variance overhead $\\gamma^2$")
    fig.tight_layout()
    out = os.path.join(LIB, "results", "error_mitigation")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"zne_diagnostic.{ext}"), dpi=160)
    print("wrote", os.path.join(out, "zne_diagnostic.pdf"))


if __name__ == "__main__":
    main()
