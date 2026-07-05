# Utility-scale shot planning

Applies `qamp_shotplanner`'s empirical-Bernstein stopping to observables drawn
from IBM's *Utility-scale quantum computing* course, to test whether adaptive
shot allocation still matters (it does — more so) at the 20–70 qubit scale.

## Observable

The transverse-field Ising model magnetization `M = (Σ_i Z_i)/N` under Lie–Trotter
time evolution (`J=1, h=-5, dt=0.1`), faithfully reproducing the `utility-ii`
notebook. `M ∈ [-1,1]` is a *bounded average*, so its per-shot variance is
suppressed as roughly `1/N` — exactly the regime the empirical Bernstein rule
exploits, since its sample complexity scales with the variance.

## Scripts

| Script | What it shows |
|---|---|
| `tfim_magnetization.py` | time sweep (reduction per evolution time) and `--scan-n` (variance-vs-N study) on the MPS simulator |
| `plot_radius_shrink.py` | radius `ε_n` shrinking across geometric checkpoints until it crosses the target `ε`, overlaying the **live `ibm_fez`** online SWAP run |
| `plot_scaling.py` | the adaptive advantage growing with system size `N` as `σ²` falls |

```bash
PYTHONPATH=src:examples/utility_scale python3.12 examples/utility_scale/tfim_magnetization.py            # time sweep, N=20
PYTHONPATH=src:examples/utility_scale python3.12 examples/utility_scale/tfim_magnetization.py --scan-n   # N = 5,10,20,40,70
PYTHONPATH=src:examples/utility_scale python3.12 examples/utility_scale/plot_radius_shrink.py
PYTHONPATH=src:examples/utility_scale python3.12 examples/utility_scale/plot_scaling.py
```

## Results (ε=0.02, δ=0.01, seed=42)

Variance-vs-N at `t=1.0` — reduction grows with scale:

| N | σ² | reduction |
|---|---|---|
| 5 | 0.305 | 1.44× |
| 10 | 0.169 | 2.33× |
| 20 | 0.090 | 4.12× |
| 40 | 0.044 | 6.03× |
| 70 | 0.027 | 8.03× |

## Honest scope

This is shot-**planning** analysis on the matrix-product-state simulator: it
reproduces the observables' structure and measures what the adaptive rule would
allocate, versus the notebooks' fixed default precision. It does **not** re-run
the hardware experiments, and the empirical-Bernstein radius bounds **sampling
error only** — orthogonal to the Trotter and hardware bias that ZNE / TREX
target (the three-level hierarchy of the thesis). Utility-scale observables are
diagonal (Z-basis, or Z after a fixed rotation), so the switch from EstimatorV2
(which hides shots behind a `precision` knob) to per-shot Sampler outcomes is
exact, not an approximation.
