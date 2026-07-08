# AdaptiveSampler — a certified counterpart to SamplerV2

`SamplerV2` draws a fixed number of shots and returns raw outcomes with no statistical
guarantee. A concentration inequality can only certify a **bounded scalar**, so
`AdaptiveSampler` names one: the probability that a shot lands in a target outcome set
`S`. Each shot becomes the Bernoulli indicator `1[outcome ∈ S] ∈ {0, 1}`, and the same
empirical-Bernstein stopper used by `AdaptiveEstimator` runs until

```
|prob − P(S)_device| ≤ ε   with probability ≥ 1 − δ,
```

reporting the shots that certificate actually cost.

## Run it

```bash
python examples/adaptive_sampler/quickstart.py
```

```python
from qiskit import QuantumCircuit
from qamp_shotplanner import AdaptiveSampler

qc = QuantumCircuit(2)
qc.ry(1.1, 0)
qc.cx(0, 1)

r = AdaptiveSampler(epsilon=0.02, delta=0.01).run([(qc, "11")])[0]
print(r.probability, r.shots)   # P('11') certified to ±0.02 w.p. ≥ 0.99
```

Pass a list of bitstrings to count several outcomes as success
(`run([(qc, ["00", "11"])])`), and a `sampler_factory` to run on hardware — the `run()`
call is unchanged.

## What it is (and isn't)

- **Is:** a certificate on a *functional* of the output distribution — a target-outcome
  probability. Useful for success-probability estimation, a marked/target bitstring
  amplitude, or any yes/no event on the measured outcome.
- **Isn't:** a certificate on the *full* output distribution. Certifying every bin of a
  distribution over `2ⁿ` outcomes scales with the support size (`~k/ε²`) and forfeits the
  variance advantage — a separate problem, out of scope here.
- **Variance-adaptive:** low- or high-probability targets have small variance and stop
  early; a target near `p = 1/2` costs the most. The quickstart prints the contrast
  (≈2,500 shots at `p ≈ 0.01` vs ≈6,600 at `p ≈ 0.5`).

This is the honest `SamplerV2` counterpart: the guarantee attaches to a bounded scalar,
exactly as it does for `AdaptiveEstimator` — see `examples/adaptive_estimator/`.
