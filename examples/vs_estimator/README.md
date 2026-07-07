# Adaptive stopping vs `EstimatorV2` default precision

A head-to-head against the tool the field actually uses. The utility-scale
notebooks (and most near-term workflows) estimate expectation values with Qiskit's
`EstimatorV2`, fixing the shot budget through `default_precision` — a target
*standard error* of `1/sqrt(n)`, so `n = ceil(1/precision^2)` shots (the default
`0.0156` → ~4096 shots). That budget is chosen **without reference to the
observable's variance**, and it is a heuristic, not a coverage guarantee: the
primitive never certifies that the estimate lands within a tolerance.

This example makes the gap empirical. For single-qubit observables `<Z> = cos θ`
spanning low to high variance, it reports the **empirical coverage**
`Pr(|Ẑ − μ| ≤ ε)` for:

- the **EstimatorV2 policy** — the fixed default-precision budget; and
- the **adaptive rule** — stop at the `(ε, δ)` Maurer–Pontil radius.

## What it shows

`EstimatorV2`'s fixed budget has **no uniform guarantee**: because it ignores
variance, it *over-covers* easy (low-variance) observables and **under-covers**
hard (high-variance) ones — its coverage swings with the observable. The adaptive
rule holds **at or above `1 − δ` everywhere**, at a variance-adaptive shot cost.

Representative run (`ε=0.02, δ=0.01`; EstimatorV2 fixed at ~4,110 shots):

| μ | σ² | EstimatorV2 coverage | adaptive coverage | adaptive mean τ |
|---|----|----|----|----|
| 0.20 | 0.960 | **83.2%** | 100.0% | 26,492 |
| 0.50 | 0.750 | **86.8%** | 100.0% | 26,492 |
| 0.80 | 0.360 | 96.5% | 100.0% | 24,414 |
| 0.95 | 0.098 | 100.0% | 100.0% | 9,788 |

EstimatorV2 under-covers the high-variance observables (83% when the target is 99%)
and over-covers the low-variance one; adaptive holds ≥ 99% throughout and spends
down to the variance (9,788 shots at μ=0.95).

This is the empirical form of the thesis's Chapter 1 / Chapter 3 argument: the
default primitive is an informal heuristic with no finite-sample coverage, and the
honest fixed alternative (Hoeffding, `n_H = 26,492`) is variance-agnostic and
wasteful — adaptive stopping is the missing middle, certifying `(ε, δ)` while
spending down to the variance.

## What it is *not*

`EstimatorV2` is not a straw man — it is the right default when you only need a
rough precision and don't require a certificate. The point is narrow and honest:
**when you need a coverage guarantee, a fixed precision does not provide one, and
its actual coverage depends on the observable.** The adaptive rule provides the
guarantee directly.

Run `python coverage_vs_estimatorv2.py`. Qiskit 2 only.
