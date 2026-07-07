"""Larger-than-H2 energy estimation: a 4-qubit multi-basis Hamiltonian.

Answers the QCE reviewer concern that the chemistry result (2-qubit H2) is small
by scaling the VQE-style energy estimation up to FOUR qubits with a genuinely
multi-basis Hamiltonian, and showing that the per-term Bonferroni + Empirical
Bernstein stopping still delivers a shot reduction with valid coverage.

What the Hamiltonian represents
-------------------------------
This is a HAND-BUILT 4-qubit, 8-term Hamiltonian standing in for a larger
molecular electronic Hamiltonian (think a 4-spin-orbital active space, or LiH /
an H2-chain reduced to 4 qubits). It is deliberately NOT taken from a real
integral computation -- the point is SCALE and measurement structure, not a
chemistry-accuracy claim. It has:

  * a constant (identity) offset -- never sampled, added exactly (as in any
    real energy estimate the identity carries no shot cost);
  * four single-qubit Z fields and a ZZ coupling -- all diagonal, read from the
    Z basis;
  * two XX exchange terms -- these do NOT commute with the Z terms and each
    needs its own rotated (X-basis) measurement circuit.

So the energy genuinely spans THREE measurement settings, exactly the
multi-basis case the thesis's Bonferroni multi-Pauli guarantee is built for
(``measured_ansatz`` supplies the per-term basis change, ``pauli_outcome_map``
the parity read-out).

Method
------
A hardware-efficient ``efficient_su2`` ansatz is lightly statevector-optimized
(multi-start COBYLA) to a near-ground state -- a stand-in for a converged VQE
solution; reproducing the optimizer is not the point. Its energy is then
estimated adaptively via ``bonferroni_estimate`` and compared to the exact
statevector energy for coverage, under two accuracy-budget splits:

  * UNIFORM ||c||_1 split (the thesis default), and
  * VARIANCE-AWARE split (eps_j ~ (sigma_j^2/|c_j|)^{1/3}), with the per-term
    standard deviations FLOORED at ``SIGMA_FLOOR``. The floor is a conservative
    variance estimate: it never lets a near-deterministic term be assigned a
    vanishing tolerance (which would inflate its shot cap), keeping the
    allocation stable. Either split preserves the joint (eps, delta) guarantee
    -- only the budget SPLIT across terms changes, not the union bound.

Everything runs on the exact statevector. Qiskit 2 only.

    python examples/larger_scale/larger_molecule.py
"""
from __future__ import annotations

from math import ceil, log

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from qamp_shotplanner import (
    bonferroni_estimate,
    measured_ansatz,
    pauli_outcome_map,
    statevector_sampler,
)

N_QUBITS = 4
EPS, DELTA = 0.02, 0.01
SIGMA_FLOOR = 0.25             # conservative per-term std floor for the var-aware split
SEEDS = (42, 43, 44, 45, 46)
TRIALS = 8                     # trials per seed

# 4-qubit illustrative "larger molecule" Hamiltonian (coeff, Pauli label).
# Labels are big-endian over qubits 3..0. Mix of diagonal (I/Z) and XX exchange.
TERMS: list[tuple[float, str]] = [
    (-1.80, "IIII"),           # constant offset (identity, never sampled)
    (0.35, "ZIII"), (0.35, "IZII"), (0.25, "IIZI"), (0.25, "IIIZ"),  # Z fields
    (0.20, "ZZII"),            # ZZ coupling (diagonal)
    (0.20, "XXII"), (0.15, "IIXX"),  # XX exchange (off-diagonal -> X basis)
]


def hamiltonian() -> SparsePauliOp:
    return SparsePauliOp([lbl for _, lbl in TERMS], [c for c, _ in TERMS])


def is_identity(label: str) -> bool:
    return set(label) == {"I"}


def prepared_state():
    """A near-ground state from a light multi-start statevector optimization.

    Stands in for a converged VQE solution; the library estimates the energy of
    a FIXED prepared state, not the outer optimization loop.
    """
    ansatz = efficient_su2(N_QUBITS, entanglement="linear", reps=1).decompose()
    H = hamiltonian()

    def energy(x: np.ndarray) -> float:
        return float(Statevector(ansatz.assign_parameters(x)).expectation_value(H).real)

    best_x, best_e = None, np.inf
    for start_seed in (7, 11, 21, 33):
        x0 = np.random.default_rng(start_seed).uniform(
            0, 2 * np.pi, ansatz.num_parameters)
        x_opt = minimize(energy, x0, method="COBYLA", options={"maxiter": 500}).x
        e = energy(x_opt)
        if e < best_e:
            best_x, best_e = x_opt, e
    return ansatz.assign_parameters(best_x), best_e


def hoeffding_baseline(sampled_terms: list[tuple[float, str]]) -> int:
    """Fixed per-term Hoeffding total at the uniform ||c||_1 split (eps_j, delta_j)."""
    m = len(sampled_terms)
    c1 = sum(abs(c) for c, _ in sampled_terms)
    eps_j = EPS / (2.0 * c1)
    delta_j = DELTA / m
    per_term = ceil((2.0**2) * log(2.0 / delta_j) / (2.0 * eps_j**2))
    return per_term * m


def per_term_sigmas(state, sampled_terms: list[tuple[float, str]]) -> list[float]:
    """Floored per-term standard deviations from the reference state (as in the
    library's simulation tables)."""
    sigmas = []
    for _, lbl in sampled_terms:
        mu = float(Statevector(state).expectation_value(SparsePauliOp(lbl)).real)
        sigma = (1.0 - mu * mu) ** 0.5 if mu * mu < 1.0 else 0.0
        sigmas.append(max(sigma, SIGMA_FLOOR))
    return sigmas


def estimate(state, sampled_terms, const, sigmas, seed_base):
    """One Bonferroni energy estimate; ``sigmas=None`` -> uniform ||c||_1 split."""
    per_term = [
        (c, statevector_sampler(measured_ansatz(state, lbl),
                                pauli_outcome_map(lbl), seed=seed_base + i))
        for i, (c, lbl) in enumerate(sampled_terms)
    ]
    res = bonferroni_estimate(per_term, eps=EPS, delta=DELTA, sigmas=sigmas)
    return res.energy + const, res.total_shots


def sweep(state, sampled_terms, const, e_ref, n_h, sigmas):
    """Mean shots / reduction / coverage over SEEDS x TRIALS for one split."""
    per_seed_tot, per_seed_cov = [], []
    for master in SEEDS:
        mrng = np.random.default_rng(master)
        tots, covs = [], []
        for _ in range(TRIALS):
            seed_base = int(mrng.integers(0, 2**31))
            energy_hat, total = estimate(state, sampled_terms, const, sigmas, seed_base)
            tots.append(total)
            covs.append(abs(energy_hat - e_ref) <= EPS / 2.0)  # energy target eps/2
        per_seed_tot.append(float(np.mean(tots)))
        per_seed_cov.append(100.0 * float(np.mean(covs)))
    tot_mean = float(np.mean(per_seed_tot))
    return {
        "tot_mean": tot_mean,
        "tot_std": float(np.std(per_seed_tot)),
        "reduction": n_h / tot_mean,
        "coverage": float(np.mean(per_seed_cov)),
    }


def main() -> None:
    H = hamiltonian()
    e_ground = float(np.linalg.eigvalsh(H.to_matrix())[0])
    sampled_terms = [(c, lbl) for c, lbl in TERMS if not is_identity(lbl)]
    const = sum(c for c, lbl in TERMS if is_identity(lbl))
    n_bases = len({"".join("Z" if p in "IZ" else p for p in lbl)
                   for _, lbl in sampled_terms})

    state, e_prepared = prepared_state()
    e_ref = float(Statevector(state).expectation_value(H).real)
    n_h = hoeffding_baseline(sampled_terms)
    sigmas = per_term_sigmas(state, sampled_terms)

    print("Larger-than-H2 energy estimation  (4 qubits, "
          f"{len(TERMS)} terms, eps={EPS}, delta={DELTA})")
    print(f"  sampled terms       : {len(sampled_terms)}  "
          f"(+1 identity offset, not sampled)")
    print(f"  measurement settings: {n_bases}  (Z group + 2 XX exchange terms)")
    print(f"  exact ground energy : {e_ground:+.4f}")
    print(f"  prepared-state E    : {e_prepared:+.4f}  (near-ground VQE stand-in)")
    print(f"  fixed Hoeffding tot : {n_h:,} shots")
    print(f"  seeds = {list(SEEDS)}, {TRIALS} trials each  (statevector, exact)\n")

    header = f"{'split':>16} {'EBS shots':>18} {'reduction':>10} {'coverage':>9}"
    print(header)
    print("-" * len(header))
    for name, sig in (("uniform ||c||_1", None), ("variance-aware", sigmas)):
        r = sweep(state, sampled_terms, const, e_ref, n_h, sig)
        print(f"{name:>16} {r['tot_mean']:>10.0f} +/- {r['tot_std']:<4.0f} "
              f"{r['reduction']:>9.2f}x {r['coverage']:>8.1f}%")

    print("\nTakeaway: on a 4-qubit multi-basis energy the adaptive Bonferroni + EBS")
    print("estimator keeps the shot budget well below the fixed Hoeffding total while")
    print("staying coverage-valid (~100%) against the exact statevector energy; the")
    print("variance-aware split concentrates shots on the high-variance terms.")


if __name__ == "__main__":
    main()
