# `AdaptiveEstimator` — a certified drop-in for `EstimatorV2`

The near-term default for expectation values is Qiskit's `EstimatorV2`: you set a
`default_precision` (a target standard error) and get back a number. That number has
**no finite-sample coverage guarantee** — the precision is a `1/√n` heuristic that
ignores the observable's variance.

`AdaptiveEstimator` keeps the same call shape but changes the contract:

```python
from qamp_shotplanner import AdaptiveEstimator

est = AdaptiveEstimator(epsilon=0.02, delta=0.01)
result = est.run([(circuit, observable)])      # observable: SparsePauliOp
r = result[0]

r.value      # certified: |value - <O>_device| <= eps  with prob >= 1 - delta
r.shots      # how many shots that certificate actually cost (adaptive)
r.epsilon, r.delta
```

It decomposes the observable into Pauli terms, estimates each with the
geometric-checkpoint empirical-Bernstein rule, and combines them by the Bonferroni
union bound (the identity term is exact and never sampled). On low-variance /
averaged observables it stops early — fewer shots, hence less QPU time — while still
certifying the tolerance.

## Same interface, any backend

The measurement backend enters only through `sampler_factory`. The default is the
exact statevector sampler, so it runs out of the box in simulation:

```python
AdaptiveEstimator(epsilon=0.02, delta=0.01)              # statevector (default)
```

To run on hardware or under noise, pass a factory built from the library's samplers
— the `run(...)` call is unchanged:

```python
from qamp_shotplanner import backend_sampler
est = AdaptiveEstimator(
    epsilon=0.02, delta=0.01,
    sampler_factory=lambda circ, omap, seed: backend_sampler(circ, omap, backend),
)
est.run([(circuit, observable)])
```

## What it is / isn't

- **Is:** a drop-in that turns "a value at some precision" into "a value with an
  `(ε,δ)` certificate and a shot count," using the library's adaptive stopping.
- **Isn't:** a mitigation layer. Like the rest of the library, it certifies the
  *sampling* error to the device-level expectation; hardware bias stays outside the
  guarantee (the three-level hierarchy).

Run `python quickstart.py` for a side-by-side against `EstimatorV2`. Qiskit 2 only.
