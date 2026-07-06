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

## From demonstration to guarantee

The example *demonstrates* that adaptive stopping composes with grouping. It also
*provably* composes: the guarantee is a one-paragraph extension of the thesis's
Bonferroni theorem, with the "items" of the union bound being **groups** instead
of single Pauli terms. The stopping rule is not modified — it is applied to a
different (grouped) observable.

**Setup.** Partition the terms `{(c_j, P_j)}`, `j = 1..m`, into `G` commuting
families `{G_1, ..., G_G}`, where every pair of terms inside a family commutes and
shares one measurement basis (single-qubit rotations for a tensor-product basis;
a Clifford rotation in general). Each family defines a **group observable**

```
O_g = sum_{j in G_g} c_j P_j ,        g = 1, ..., G,
```

read from ONE circuit: rotate into the family's basis, measure once, map each
shot's bitstring to the summed eigenvalue `O_g in [-R_g/2, +R_g/2]` with
`R_g <= 2 * sum_{j in G_g} |c_j|` (tighter if the spectrum is known). The energy
is `E = sum_g <O_g>`.

**Proposition (commuting-family EBS guarantee).** Estimate each `<O_g>` with the
thesis's geometric-checkpoint (or anytime) empirical-Bernstein rule at range
`R_g`, tolerance `eps_g`, and budget `delta_g = delta / G`. If
`sum_g eps_g <= eps`, then with probability at least `1 - delta`,

```
| E_hat - E_device |  <=  sum_g eps_g  <=  eps ,     E_hat = sum_g mu_hat_g .
```

**Proof sketch.** By the single-observable Maurer-Pontil guarantee, each group
estimate is `(eps_g, delta_g)`-correct: `Pr(|mu_hat_g - <O_g>| > eps_g) <= delta_g`
— that guarantee is stated for *any* bounded observable, and `O_g` is one, its
range entering only through `R_g`. A union bound over the `G` groups gives
`Pr(any group fails) <= sum_g delta_g = delta`. On the complement, the triangle
inequality yields `|E_hat - E| <= sum_g eps_g`. This is exactly the thesis's
Bonferroni proof with `m -> G` and single Paulis -> group observables.

Three facts make that substitution free:

1. **Range is a parameter.** The Maurer radius
   `sqrt(2 V_n log(4/delta)/n) + 7 R log(4/delta)/(3(n-1))` takes `R = R_g`, so a
   wide group observable needs no new inequality.
2. **Correlations are absorbed.** Terms inside a group are measured from the same
   shots and are correlated, but the rule estimates `Var(O_g)` *empirically*;
   empirical Bernstein adapts to that variance directly, so within-group
   covariances require no independence assumption — they are already inside `V_n`.
3. **The union bound is item-agnostic.** Bonferroni does not distinguish `m`
   single terms from `G` groups; only `sum (tolerance) <= eps` and
   `sum (budget) = delta` matter.

**Why it lifts the limit.** The cost now scales with the number of *groups* `G`
and the group variances `Var(O_g)`, not with `||c||_1^2`. Two levers: `G << m`
shrinks both the tolerance split and the `delta/G` budget (Kagome: 3 vs 54), and
shots are *shared* across the terms in a family, with EBS stopping early wherever
`Var(O_g)` is small (as it is near a structured / ground state).

## Roadmap (v0.3.0)

Turning the demonstration into a first-class feature:

1. **Grouping front-end.** Accept a partition into commuting families. Start with
   tensor-product-basis grouping (qubit-wise-commuting terms, single-qubit
   rotations only — what this example does by hand), then general commuting groups
   via a Clifford diagonalizing rotation, and graph-colouring on the commutation
   graph to minimise `G`.
2. **Group-measurement sampler.** Generalise `measured_ansatz` +
   `statevector_value_sampler` (added here) into a reusable group sampler, plus a
   hardware counterpart that records the per-shot summed eigenvalue.
3. **Group-level Bonferroni.** Extend `bonferroni_estimate` to take groups, each
   with its own range `R_g` and sampler, and split `eps` over groups — uniform,
   tight, or **variance-aware over groups** `eps_g ~ Var(O_g)^(1/3)`, mirroring the
   per-term allocation the library already has.
4. **State the proposition.** Promote the sketch above to a proposition in the
   methodology, with the union-bound proof — a short, self-contained extension of
   the existing Bonferroni theorem.

**Honest scope.** Grouping removes the `||c||_1^2` *overhead*; it does not beat the
information-theoretic floor (the Le Cam bound), it moves closer to it. General
(non-tensor-product) groups cost extra circuit depth for the Clifford rotation — a
real trade-off. The open question v0.3.0 raises is choosing the grouping *and* the
per-group allocation that jointly minimise total shots under the `(eps, delta)`
guarantee.

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
