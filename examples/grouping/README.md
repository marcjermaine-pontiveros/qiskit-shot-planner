# Grouped estimation — commuting-set measurement grouping

The general form of the hand-built 3-group Kagome example
(`examples/kagome_heisenberg`). It answers the reviewer question: *how does the
shot-planning strategy operate under Qiskit-style commuting-set measurement
grouping, and do the `(eps, delta)` guarantees still hold?*

## What qubit-wise-commuting (QWC) grouping is

Two Pauli terms are **qubit-wise commuting** iff on every qubit their single-qubit
factors are equal or at least one is the identity. Such terms can be read
*simultaneously* from a single **tensor-product measurement basis** (single-qubit
rotations only): rotate each qubit into the group's letter, measure once in Z, and
every term in the group is evaluated from that same shot.

`qubitwise_commuting_groups(terms)` greedily partitions `(coeff, label)` Pauli
terms into QWC families (first-fit; deterministic and order-stable). Each
`QWCGroup` carries its terms and the merged measurement `basis` (the non-identity
letter per qubit, big-endian like Qiskit Pauli strings).

`grouped_energy_estimate(terms, state_circuit, eps, delta, sampler_factory)` then
estimates `E = Σ_j c_j ⟨P_j⟩` by running one Empirical-Bernstein stopper **per
group** on the group observable `O_g = Σ_{j∈G_g} c_j P_j` (range
`R_g = 2 Σ_{j∈G_g} |c_j|`), and combines the group means.

## The guarantee — union bound over groups

The `(eps, delta)` guarantee is **unchanged**; only the union-bound "items" become
groups instead of single Pauli terms:

- Estimate each `⟨O_g⟩` with the thesis's Maurer–Pontil empirical-Bernstein rule at
  range `R_g`, tolerance `eps_g`, budget `delta_g = delta / G`.
- If `Σ_g eps_g ≤ eps`, then with probability `≥ 1 − delta`,
  `|E_hat − E| ≤ Σ_g eps_g ≤ eps`, with `E_hat = Σ_g μ̂_g`.

This is the thesis Bonferroni theorem with `m → G` and single Paulis → group
observables. Three facts make the substitution free: range enters only as the
parameter `R_g`; within-group correlations are absorbed because empirical Bernstein
adapts to `Var(O_g)` directly; and the union bound is item-agnostic. See the full
**"commuting-family EBS guarantee" proposition and proof** in
`examples/kagome_heisenberg/README.md`.

The tolerance split (`split=`) keeps `Σ_g eps_g = eps` either way:
`"uniform"` gives `eps_g = eps/G`; `"tight"` gives `eps_g ∝ R_g^{2/3}` (the
range-aware allocation that minimises the Hoeffding leading cost, a variance-free
proxy that needs no pilot pass).

## The demo

`python mixed_hamiltonian.py` runs a transverse-field Ising model
`H = J·Σ ZᵢZᵢ₊₁ + h·Σ Xᵢ` — a Hamiltonian with a **mix** of commuting and
non-commuting terms. The `m = (N−1) + N` Pauli terms collapse into exactly **two**
QWC groups (all-Z couplings, all-X fields, which do not mutually commute). It prints
the naive per-term count and Hoeffding wall vs the two-group grouped estimate, and
checks the energy lands within `eps` of the exact value at the `(eps, delta)`
guarantee.

## What it uses

- `qubitwise_commuting_groups`, `grouped_energy_estimate` (new in
  `planners/grouped.py`).
- `statevector_value_sampler` — the general multi-valued statevector sampler.
- `EmpiricalBernsteinStopper` / `HoeffdingPlanner` from the core library.

Qiskit 2 only.
