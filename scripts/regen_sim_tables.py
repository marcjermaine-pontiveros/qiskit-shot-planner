"""Regenerate the simulation tables over multiple master seeds (mean ± std).

Fills thesis Chapter 5:
  - Table 5.2 (sample complexity vs variance)  -> sample_complexity.csv
  - Table 5.4 lower-bound column (le_cam_two_point)  -> same file
  - Table 5.1 per-workload summary (SWAP / QAOA / VQE)  -> workloads.csv

Everything is exact-statevector or synthetic Bernoulli (no Aer, no hardware), so
it is cheap and fully reproducible. Run:

    PYTHONPATH=src python3 scripts/regen_sim_tables.py
"""

from __future__ import annotations

import argparse
import csv
import os
from math import cos, sqrt

import numpy as np

from qamp_shotplanner import (
    AnytimeEBSStopper,
    EmpiricalBernsteinStopper,
    HoeffdingPlanner,
    bonferroni_estimate,
    h2_terms,
    ideal_zz,
    le_cam_two_point,
    pauli_outcome_map,
    qaoa_maxcut_circuit,
    statevector_sampler,
    swap_test_1qubit,
    vqe_ansatz,
    measured_ansatz,
    zz_outcome_map,
)

THESIS_SEEDS = [42, 43, 44, 45, 46]  # multi-seed: mean +/- std for the thesis
PCSC_SEED = [42]              # single seed: reproduces the published PCSC numbers
EPS, DELTA = 0.02, 0.01
SIGMA2_GRID = [0.00, 0.05, 0.25, 0.50, 0.75, 1.00]
THETA1, THETA2 = 0.3, 0.8
GAMMA, BETA = 0.783, 0.438
H2_BOND = 0.74


def _ancilla_outcome(bitstring: str) -> float:
    return 1.0 if bitstring[-1] == "0" else -1.0


def _bernoulli_device(mu: float, seed: int):
    """Synthetic ±1 sampler with exact mean mu (variance 1 - mu**2)."""
    p = 0.5 * (1.0 + mu)
    rng = np.random.default_rng(seed)
    return lambda n: (2.0 * rng.binomial(1, p, int(n)) - 1.0).tolist()


def _ebs(sampler):
    return EmpiricalBernsteinStopper(
        epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
    ).run_batched(sampler)


def _anytime(sampler):
    return AnytimeEBSStopper(
        epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0
    ).run_batched(sampler)


def _agg(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _write(name: str, rows: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def variance_sweep(trials: int, seeds: list[int]) -> list[dict]:
    """EBS-geom vs Hoeffding across the variance grid, over SEEDS."""
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    rows = []
    for s2 in SIGMA2_GRID:
        mu = sqrt(max(0.0, 1.0 - s2))
        per_seed_tau, per_seed_cov = [], []
        for master in seeds:
            mrng = np.random.default_rng(master)
            taus, cov = [], []
            for _ in range(trials):
                sd = int(mrng.integers(0, 2**31))
                r = _ebs(_bernoulli_device(mu, sd))
                taus.append(r.n)
                cov.append(abs(r.estimate - mu) <= EPS)
            per_seed_tau.append(float(np.mean(taus)))
            per_seed_cov.append(100.0 * float(np.mean(cov)))
        tau_m, tau_s = _agg(per_seed_tau)
        cov_m, cov_s = _agg(per_seed_cov)
        lb = le_cam_two_point(EPS, DELTA, s2)
        print(f"  σ²={s2:.2f}  E[τ]={tau_m:8.0f}±{tau_s:.0f}  "
              f"reduction={n_h / tau_m:.2f}×  coverage={cov_m:.1f}±{cov_s:.1f}%  LB={lb:.0f}")
        rows.append({
            "sigma2": f"{s2:.2f}",
            "mu": f"{mu:.4f}",
            "n_hoeffding": n_h,
            "ebs_tau_mean": f"{tau_m:.1f}",
            "ebs_tau_std": f"{tau_s:.1f}",
            "reduction": f"{n_h / tau_m:.3f}",
            "coverage_mean_pct": f"{cov_m:.2f}",
            "coverage_std_pct": f"{cov_s:.2f}",
            "le_cam_lb": f"{lb:.1f}",
            "ebs_over_lb": f"{tau_m / lb:.2f}",
        })
    return rows


def _workload_row(name: str, sigma2: float, ideal: float, make_sampler, trials: int,
                  n_h: int, seeds: list[int]) -> dict:
    """Run EBS-geom + anytime for a statevector workload over SEEDS."""
    ebs_seed_tau, ebs_seed_cov, any_seed_tau = [], [], []
    for master in seeds:
        mrng = np.random.default_rng(master)
        e_tau, e_cov, a_tau = [], [], []
        for _ in range(trials):
            sd = int(mrng.integers(0, 2**31))
            r = _ebs(make_sampler(sd))
            e_tau.append(r.n)
            e_cov.append(abs(r.estimate - ideal) <= EPS)
            ra = _anytime(make_sampler(sd + 7))
            a_tau.append(ra.n)
        ebs_seed_tau.append(float(np.mean(e_tau)))
        ebs_seed_cov.append(100.0 * float(np.mean(e_cov)))
        any_seed_tau.append(float(np.mean(a_tau)))
    et_m, et_s = _agg(ebs_seed_tau)
    ec_m, ec_s = _agg(ebs_seed_cov)
    at_m, _ = _agg(any_seed_tau)
    print(f"  {name:5s} σ²={sigma2:.4f}  EBS τ={et_m:.0f}±{et_s:.0f} ({n_h / et_m:.2f}×)  "
          f"anytime τ={at_m:.0f} ({n_h / at_m:.2f}×)  coverage={ec_m:.1f}%")
    return {
        "workload": name,
        "sigma2": f"{sigma2:.4f}",
        "n_hoeffding": n_h,
        "ebs_tau_mean": f"{et_m:.1f}",
        "ebs_tau_std": f"{et_s:.1f}",
        "ebs_reduction": f"{n_h / et_m:.3f}",
        "anytime_tau_mean": f"{at_m:.1f}",
        "anytime_reduction": f"{n_h / at_m:.3f}",
        "coverage_mean_pct": f"{ec_m:.2f}",
        "coverage_std_pct": f"{ec_s:.2f}",
    }


def workloads(trials: int, seeds: list[int]) -> list[dict]:
    n_h = HoeffdingPlanner(epsilon_stat=EPS, delta=DELTA, a=-1.0, b=1.0).planned_shots()
    rows = []

    swap_qc = swap_test_1qubit(THETA1, THETA2)
    f_ideal = cos((THETA1 - THETA2) / 2.0) ** 2
    rows.append(_workload_row(
        "SWAP", 1.0 - f_ideal**2, f_ideal,
        lambda s: statevector_sampler(swap_qc, _ancilla_outcome, seed=s), trials, n_h, seeds))

    qaoa_qc = qaoa_maxcut_circuit(GAMMA, BETA)
    zz = ideal_zz(GAMMA, BETA)
    rows.append(_workload_row(
        "QAOA", 1.0 - zz**2, zz,
        lambda s: statevector_sampler(qaoa_qc, zz_outcome_map, seed=s), trials, n_h, seeds))

    # VQE: Bonferroni over the H2 Pauli terms at equilibrium, using the
    # variance-aware allocation of the manuscript; reduction vs the fixed
    # uniform-split per-term Hoeffding total (the same baseline the manuscript
    # reports for both splits).
    terms = h2_terms(H2_BOND)
    ansatz = vqe_ansatz(*_h2_angles())
    n_h_vqe = _vqe_hoeffding_total(terms)  # fixed baseline: uniform eps_j=eps/(2|c|_1)
    from qiskit.quantum_info import Statevector, SparsePauliOp
    sv = Statevector(ansatz)
    mus = [float(sv.expectation_value(SparsePauliOp(lbl)).real) for _, lbl in terms]
    e_ref = float(sum(c * mu for (c, _), mu in zip(terms, mus)))
    # Per-term standard deviations of the +/-1 observables drive the
    # variance-aware split eps_j ~ (sigma_j^2/|c_j|)^{1/3} (App. B multi-LB).
    sigmas = [(1.0 - mu * mu) ** 0.5 if mu * mu < 1.0 else 0.0 for mu in mus]
    vqe_seed_tau, vqe_seed_cov = [], []
    for master in seeds:
        mrng = np.random.default_rng(master)
        tots, covs = [], []
        for _ in range(trials):
            sd = int(mrng.integers(0, 2**31))
            per_term = [
                (c, statevector_sampler(measured_ansatz(ansatz, lbl), pauli_outcome_map(lbl), seed=sd + i))
                for i, (c, lbl) in enumerate(terms)
            ]
            res = bonferroni_estimate(per_term, eps=EPS, delta=DELTA, sigmas=sigmas)
            tots.append(res.total_shots)
            covs.append(abs(res.energy - e_ref) <= EPS / 2.0)  # energy-accurate target eps/2
        vqe_seed_tau.append(float(np.mean(tots)))
        vqe_seed_cov.append(100.0 * float(np.mean(covs)))
    vt_m, vt_s = _agg(vqe_seed_tau)
    vc_m, vc_s = _agg(vqe_seed_cov)
    print(f"  VQE  bond={H2_BOND}A  E_ref={e_ref:.4f}  EBS total tau={vt_m:.0f}+/-{vt_s:.0f} "
          f"({n_h_vqe / vt_m:.2f}x)  coverage={vc_m:.1f}%  ({len(terms)} terms)")
    rows.append({
        "workload": "VQE", "sigma2": "n/a (multi-term)", "n_hoeffding": n_h_vqe,
        "ebs_tau_mean": f"{vt_m:.1f}", "ebs_tau_std": f"{vt_s:.1f}",
        "ebs_reduction": f"{n_h_vqe / vt_m:.3f}", "anytime_tau_mean": "n/a",
        "anytime_reduction": "n/a", "coverage_mean_pct": f"{vc_m:.2f}", "coverage_std_pct": f"{vc_s:.2f}",
    })
    return rows


def _h2_angles():
    from qamp_shotplanner import H2_ANGLES
    return H2_ANGLES[H2_BOND]


def _vqe_hoeffding_total(terms: list[tuple[float, str]]) -> int:
    """Hoeffding total at the SAME per-term split Bonferroni uses (eps_j, delta_j)."""
    from math import ceil, log
    m = len(terms)
    c1 = sum(abs(c) for c, _ in terms)
    eps_j = EPS / (2.0 * c1)
    delta_j = DELTA / m
    per_term = ceil((2.0**2) * log(2.0 / delta_j) / (2.0 * eps_j**2))
    return per_term * m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=500, help="trials per seed (sweep)")
    ap.add_argument("--wl-trials", type=int, default=200, help="trials per seed (workloads)")
    ap.add_argument("--seeds", type=int, nargs="+", default=THESIS_SEEDS,
                    help="master seeds; pass a single 42 for the PCSC-reproduction version")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    print(f"Regen sim tables: ε={EPS}, δ={DELTA}, seeds={args.seeds}")
    print("Variance sweep (EBS-geom vs Hoeffding):")
    vs = variance_sweep(args.trials, args.seeds)
    print(_write("sample_complexity.csv", vs, args.out))

    print("Workloads (SWAP / QAOA / VQE):")
    wl = workloads(args.wl_trials, args.seeds)
    print(_write("workloads.csv", wl, args.out))


if __name__ == "__main__":
    main()
