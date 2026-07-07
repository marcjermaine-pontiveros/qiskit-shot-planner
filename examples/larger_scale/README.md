# Larger-scale simulation benchmarks

These two examples answer the QCE reviewer concern (W1.1 / W2.1) that every
reported experiment is small (<= 3 qubits): "test a larger chemistry application
or a larger QAOA instance." They show, in exact statevector simulation, that the
adaptive Empirical Bernstein stopping **still works and stays coverage-valid** at
larger sizes, and characterize how the shot reduction behaves as the system
grows.

Both are pure statevector runs (no Aer, no hardware), so they are cheap and fully
reproducible.

```bash
python examples/larger_scale/qaoa_scaling.py     # ~1 min
python examples/larger_scale/larger_molecule.py  # ~1 min
```

## 1. `qaoa_scaling.py` — QAOA MaxCut up to 14 qubits

A p=1 QAOA MaxCut instance on a circulant ring-plus-chords graph (edges
`(i, i+1)` and `(i, i+2)` mod N, 4-regular, `|E| = 2N`), swept over
**N = 6, 10, 14 qubits** with fixed angles `gamma = 0.5, beta = 0.2`.

All `Z_i Z_j` cost terms commute, so the whole cost is read from ONE Z-basis
circuit; we estimate the intensive cost density `e = C/|E| in [-1, 1]` as a single
bounded observable with the EBS stopper.

| qubits | edges | e_exact | variance | EBS shots | reduction | coverage |
|-------:|------:|--------:|---------:|----------:|----------:|---------:|
| 6      | 12    | +0.1135 | 0.0810   | ~8,557    | 3.10x     | 100.0%   |
| 10     | 20    | +0.1089 | 0.0393   | ~6,365    | 4.16x     | 100.0%   |
| 14     | 28    | +0.1089 | 0.0280   | ~5,314    | 4.99x     | 100.0%   |

Fixed Hoeffding budget for the same `(eps=0.02, delta=0.01)`: **26,492 shots**
(5 seeds x 20 trials each).

**Finding:** as the graph grows, more edges are averaged into the density, its
per-shot variance falls, and the adaptive advantage over the fixed Hoeffding
budget **grows with system size** (3.1x -> 5.0x) — while coverage against the
exact statevector cost stays at 100%.

## 2. `larger_molecule.py` — a 4-qubit multi-basis energy

A hand-built **4-qubit, 8-term** Hamiltonian standing in for a larger molecular
electronic Hamiltonian (e.g. a 4-spin-orbital active space / LiH-scale
reduction). It has a constant identity offset (never sampled), four single-qubit
Z fields, a ZZ coupling, and two XX exchange terms — so the energy genuinely
spans **three measurement settings** (a Z group plus two X-basis exchange terms),
the multi-basis case the thesis's Bonferroni multi-Pauli guarantee is built for.

A hardware-efficient `efficient_su2` ansatz is lightly statevector-optimized
(multi-start COBYLA) to a near-ground state (E = -2.8498 vs exact ground -2.8500)
as a converged-VQE stand-in, then its energy is estimated adaptively with
`bonferroni_estimate`.

| split            | EBS shots  | reduction | coverage |
|------------------|-----------:|----------:|---------:|
| uniform `||c||_1` | ~1,236,000 | 2.51x     | 100.0%   |
| variance-aware   | ~230,000   | 13.48x    | 100.0%   |

Fixed Hoeffding total for the same `(eps=0.02, delta=0.01)`: **3,105,963 shots**
(5 seeds x 8 trials each; energy target `eps/2`).

The variance-aware split allocates `eps_j ~ (sigma_j^2 / |c_j|)^{1/3}` with the
per-term standard deviations floored at 0.25 (a conservative estimate that keeps
the allocation stable). Either split preserves the joint `(eps, delta)`
guarantee — only the budget SPLIT across terms changes, not the union bound —
which is why coverage stays at 100% for both.

**Finding:** on a 4-qubit multi-basis energy the adaptive estimator keeps the
shot budget 2.5x–13.5x below the fixed Hoeffding total while staying
coverage-valid against the exact statevector energy; the variance-aware split
concentrates shots on the high-variance terms.

## Honest caveats

- These are **statevector-simulation scale demonstrations** (N <= 16 qubits,
  noiseless, exact reference). They show the adaptive stopping and its coverage
  guarantee remain effective as the observable / graph grows; they are **not**
  claims about hardware-scale performance or chemistry accuracy.
- The 4-qubit Hamiltonian is **illustrative**, not from a real integral
  computation — it is chosen to exercise the multi-basis measurement structure
  at a larger size, not to reproduce a molecule's energy.
- Running the estimator at hardware scale (deeper circuits, device noise, error
  mitigation, real chemistry Hamiltonians) is separate future work; the online
  orchestration path (`examples/utility_scale/`, the live `ibm_fez` runs) is the
  hardware-facing companion to these noiseless scale checks.
- Numbers vary by <1% run-to-run (the per-seed std columns quantify this); the
  reductions and 100% coverage are stable.
