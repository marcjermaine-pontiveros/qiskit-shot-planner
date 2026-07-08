# Certified drop-ins vs. Qiskit V2 primitives — simulator and hardware

`adaptive_vs_v2_primitives.ipynb` runs all four side by side on one workload:

| | Returns | Guarantee |
|---|---|---|
| `EstimatorV2` | ⟨O⟩ at fixed `precision` | none (a 1σ target, not coverage) |
| `AdaptiveEstimator` | ⟨O⟩ | `\|value − ⟨O⟩\| ≤ ε` w.p. `≥ 1 − δ` |
| `SamplerV2` | raw counts at fixed `shots` | none |
| `AdaptiveSampler` | `P(outcome ∈ S)` | `\|prob − P(S)\| ≤ ε` w.p. `≥ 1 − δ` |

## What it shows

1. **`EstimatorV2` vs `AdaptiveEstimator`** on `AerSimulator`. In the executed run,
   `EstimatorV2(precision=0.02)` returned a value **0.022 off** the exact answer — *outside*
   its nominal precision, because `precision` is a standard-error target, not a coverage
   guarantee. `AdaptiveEstimator` landed within ε and reported the shots that certificate cost.
2. **`SamplerV2` vs `AdaptiveSampler`** — a fixed-shot distribution with a hand-computed
   `P(target)` vs a certified target-outcome probability, plus the variance-adaptivity of the
   stopping rule (a `p ≈ 0.01` target stops in far fewer shots than `p ≈ 0.5`).
3. **The same code on real hardware.** Both drop-ins take the backend only through a
   `sampler_factory`; `backend_sampler` routes each adaptive batch to `SamplerV2` for IBM
   backends or `backend.run` for `AerSimulator`. Swapping `AerSimulator()` for an `IBMBackend`
   is the only change — the hardware cell is guarded by `RUN_HARDWARE = False` so the notebook
   never submits a live job by accident.

## Run it

```bash
jupyter notebook examples/primitives_comparison/adaptive_vs_v2_primitives.ipynb
```

The notebook ships with executed outputs. To run the hardware section, configure
`QiskitRuntimeService` credentials and set `RUN_HARDWARE = True`.

**Scope (honest):** the certificate is on *sampling* error relative to the **device**
expectation, not on hardware bias relative to the ideal value — see the three-level error
hierarchy in the docs.
