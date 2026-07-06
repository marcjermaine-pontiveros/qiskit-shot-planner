# PCSC 2026 reproducibility scripts

Seed-pinned drivers that regenerate the PCSC / QCE / thesis numbers using the
productized `qamp_shotplanner` API — no inline EBS/Hoeffding/QAOA/noise code.
Each script builds a workload circuit + outcome map, wires a sampler, runs the
three stopping rules (Hoeffding, EBS-geom, Anytime-EBS), prints a summary, and
writes a CSV under `results/`.

Every script uses **`SEED = 42`** and the anchor accuracy target
**ε = 0.02, δ = 0.01** (the Hoeffding budget for a ±1 observable is then
`n_H = ⌈2·ln(2/δ)/ε²⌉ = 26492`). `repro_coverage.py` also sweeps ε = 0.05.

## Running

The repo's `.venv` pins Python 3.14, where `qiskit-aer` fails to build. Use a
Python 3.12 interpreter with `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`,
`numpy`, `scipy` installed, and put the package on the path via `PYTHONPATH=src`:

```bash
# from the repo root
PYTHONPATH=src python3 examples/pcsc2026/repro_qaoa.py
PYTHONPATH=src python3 examples/pcsc2026/repro_swap.py
PYTHONPATH=src python3 examples/pcsc2026/repro_coverage.py
PYTHONPATH=src python3 examples/pcsc2026/repro_noise_sweep.py
PYTHONPATH=src python3 examples/pcsc2026/repro_zne.py
PYTHONPATH=src python3 examples/pcsc2026/repro_vqe_h2.py
```

Each script takes `--trials N` (default = the paper trial count) and `--out DIR`
(default `examples/pcsc2026/results/`). Pass a small `--trials` for a quick
check; the defaults reproduce the reported statistics. `repro_noise_sweep.py`
and `repro_zne.py` are the heaviest (each trial pre-simulates an Aer shot
buffer) — `repro_zne.py` also accepts `--shots`.

## Noise models

The noisy scripts use the library noise models in `backends/noise_models.py`,
fed to `noise_model_sampler` (buffered AerSimulator). Parameters:

| Model | Channels | Used by |
|-------|----------|---------|
| `depolarizing_noise_model(p1, p2)` | 1q depolarizing `p1` on `h`,`rx`; 2q depolarizing `p2` on `cx` | `repro_noise_sweep.py` (`p1=p/10, p2=p`, p ∈ {0, 0.01, …, 0.50}); `repro_zne.py` (`p1=0.005, p2=0.05`) |
| `calibrated_noise_model(p, readout)` | 1q depolarizing `p`; 2q depolarizing `min(5p, 1)`; symmetric readout flip `readout` | available (QCE exp2 tied `readout = min(0.5p, 0.5)`) |
| `fake_montreal_simulator()` | `AerSimulator.from_backend(FakeMontrealV2())` — recorded 27-qubit IBM Montreal T1/T2, gate, and readout calibration | available (exp3 realistic backend) |

`repro_coverage.py` uses an analytic depolarizing device model μ = (1−p)²
(Bernoulli ±1, p₊ = (1+μ)/2) rather than an Aer simulation — this is the
closed-form SWAP-test channel used to validate coverage cheaply over 500 trials.

## What each script regenerates

| Script | Workload / sampler | Regenerates |
|--------|--------------------|-------------|
| `repro_swap.py` | SWAP test θ₁=0.3, θ₂=0.8; `statevector_sampler` on the ancilla | Thesis SWAP worked example: F ≈ 0.939, σ² ≈ 0.12, n_H = 26492, EBS-geom ≈ 3.4× at (ε=0.02, δ=0.01) |
| `repro_qaoa.py` | QAOA MaxCut γ=0.783, β=0.438; `statevector_sampler` on ⟨ZZ⟩ | QAOA energy-estimation table: ⟨ZZ⟩ ≈ 0.984, σ² ≈ 0.032, EBS-geom ≈ 8×, Anytime ≈ 1.7× |
| `repro_noise_sweep.py` | QAOA ⟨ZZ⟩; `noise_model_sampler` + `depolarizing_noise_model` sweep | Shot-savings-vs-noise result: EBS ≈ 7× at p=0 degrading to 1.0× (cap) as σ² → 1; Anytime reaches the cap by p ≈ 0.02 |
| `repro_coverage.py` | Analytic μ=(1−p)² device; (ε, δ, p) grid, 500 trials | Empirical coverage validation (Prop. 5.x): all rules hold coverage ≥ 1−δ = 99% |
| `repro_zne.py` | Bell ⟨ZZ⟩ under depolarizing p=0.05; `fold_gates` + `zne_extrapolate` | ZNE + adaptive-stopping complementarity: gate-folding bias curve, linear/exp extrapolation to zero noise, per-factor EBS shot savings |
| `repro_vqe_h2.py` | H2 STO-3G ansatz; `bonferroni_estimate` over 4 Pauli terms, bond-length sweep | Multi-observable VQE energy: |ΔE| within ε, EBS-geom total-shot reduction ≈ 2–5.7× vs. the matched Bonferroni Hoeffding baseline |

Notes:
- `repro_vqe_h2.py` is a **statevector (noiseless) simulation** reproduction of
  the H2 energy estimate; it is not the planned hardware (QCE) validation.
- `bonferroni_estimate` refines the flat per-term ε used in the source QCE
  experiment to the thesis ‖c‖₁ split (ε_j = ε/(2‖c‖₁), δ_j = δ/m); the
  Hoeffding baseline in `repro_vqe_h2.py` uses the same split for a like-for-like
  reduction ratio.
- Anytime-EBS uses the π² pointwise schedule δ_n = 6δ/(π²n²); it is more
  conservative at large n and degrades to the Hoeffding cap for
  moderate-variance observables, as reported.
