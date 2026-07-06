# Kagome Heisenberg — the limit of per-term Bonferroni

An exploratory example (post-v0.2.0) motivated by the IBM Open Science Prize 2022
Kagome VQE problem, reframed as a shot-allocation study. It is deliberately a
**limits** example: it shows where the thesis's per-term Bonferroni guarantee
stops being practical, and the change of measurement strategy that lifts it.

## The system

The 12-site Kagome unit cell Heisenberg Hamiltonian (18 edges):

```
H = sum_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)      ->  54 Pauli terms
```

## The limit

Per-term Bonferroni splits the energy tolerance as `eps_j = eps / (2||c||_1)`
with `||c||_1 = 54`. Each of the 54 terms must therefore be resolved about
`||c||_1^2 ≈ 2900x` more tightly than a single observable, so the per-term
Hoeffding cap explodes:

```
eps_j = 0.02 / 108 ≈ 1.85e-4   ->   per-term cap ≈ 5.4e8 shots  (≈ 2.9e10 total)
```

That is not simulable and not runnable on hardware. The `||c||_1^2` penalty is
the honest scaling limit of the grouping-free multi-Pauli method as the
Hamiltonian grows.

## The resolution

The 54 terms fall into **3 commuting families** (all-X, all-Y, all-Z), each read
from a single basis circuit so the 18 edges in a family **share shots**. We
estimate the **intensive energy density** `e = E/N` (a bounded `[-1.5, 1.5]`
observable), for which the fixed budget is modest and EBS gives a clean
reduction over Hoeffding — with the union bound now over 3 groups, not 54 terms.

Running `python kagome_energy.py` prints the naive per-term wall, then the
grouped/intensive run (a ~10x EBS reduction at exact coverage).

## What it uses

- `statevector_value_sampler` — the general multi-valued statevector sampler
  (v0.2.x): a per-shot group value takes more than two levels, unlike the ±1
  `statevector_sampler`.
- `EmpiricalBernsteinStopper` / `HoeffdingPlanner` from the core library.

## What it does *not* do

- It does not reproduce the SPSA/VQE optimization (the outer loop is orthogonal
  to shot allocation); it fixes a structured low-energy state and estimates it.
- It does not showcase **variance-aware** allocation: the 3 families are
  near-symmetric, so uniform ≈ variance-aware here. The variance-aware win is
  the H₂ VQE example; this example's lesson is grouping + intensive scaling.

Qiskit 2 only.
