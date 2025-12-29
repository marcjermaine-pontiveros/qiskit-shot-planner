# QAMP Shot Planner

Budgeting circuit runs for expectation values using concentration bounds.

## Overview

This package provides tools for planning the number of shots needed to estimate quantum circuit expectation values with statistical guarantees. It implements Hoeffding's inequality for bounded random variables and includes a complete SWAP test fidelity estimation workflow.

## MVP Features

### What's Already Implemented

- **Hoeffding Planner**: Generic shot planning for bounded variables X ∈ [a, b]
- **SWAP Test Circuits**: 3-qubit SWAP test for single-qubit state fidelity estimation
- **EstimatorV2 Integration**: Qiskit Aer EstimatorV2 execution wrapper with noise modeling
- **Coverage Validation**: Empirical validation of Hoeffding bounds through repeated trials
- **Demo Notebook**: Reproducible end-to-end workflow with headline results (26,492 shots for εF=0.02, δ=0.01)

### Achieved Results

From the experimental validation:
- **0/1000 bound violations** in coverage validation (exceeds theoretical δ=0.01)
- **Three-level error decomposition**: Statistical error vs. hardware bias vs. total error
- **Reproducible demo**: Consistent results across multiple runs with deterministic seeds

## Installation

```bash
# Install in editable mode
uv pip install -e .

# Or with pip
pip install -e .
```

## Quickstart

```python
import math
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qamp_shotplanner import (
    plan_shots_for_swap_fidelity,
    swap_test_1qubit,
    run_swap_fidelity_estimator,
)

# 1. Plan shots using Hoeffding bound
epsilon_F = 0.02  # tolerance on overlap² error
delta = 0.01      # failure probability
shots = plan_shots_for_swap_fidelity(epsilon_F, delta)
print(f"Planned shots: {shots}")  # 26,492

# 2. Build SWAP test circuit
theta1, theta2 = 0.3, 0.8
qc = swap_test_1qubit(theta1, theta2)

# 3. Run with optional noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 1),
    ["ry", "h"],
)

overlap2_hat = run_swap_fidelity_estimator(
    qc,
    shots=shots,
    noise_model=noise_model,
    seed_simulator=42,
)

print(f"Estimated overlap²: {overlap2_hat:.4f}")
```

## Public API

### Core Planner

```python
from qamp_shotplanner import HoeffdingPlanner

planner = HoeffdingPlanner(
    epsilon_stat=0.02,  # tolerance on |hat(μ) - μ|
    delta=0.01,          # failure probability
    a=-1.0,              # lower bound
    b=1.0                # upper bound
)
shots = planner.planned_shots()
```

### SWAP Test Planning

```python
from qamp_shotplanner import plan_shots_for_swap_fidelity

# Plans shots for overlap² estimation
# Hoeffding bounds apply to per-shot Z_anc ∈ [-1, 1]
shots = plan_shots_for_swap_fidelity(epsilon_F=0.02, delta=0.01)
```

### SWAP Circuit

```python
from qamp_shotplanner import swap_test_1qubit

qc = swap_test_1qubit(theta1=0.3, theta2=0.8)
# Returns: 3-qubit circuit (ancilla at index 0)
# - Per-shot: Z_anc ∈ {+1, -1}
# - Expected value: E[Z_anc] = overlap² ∈ [0, 1]
```

### Estimator Execution

```python
from qamp_shotplanner import run_swap_fidelity_estimator

overlap2_hat = run_swap_fidelity_estimator(
    qc,
    shots=26492,
    noise_model=None,  # optional Qiskit noise model
    seed_simulator=42  # optional seed for reproducibility
)
# Returns: overlap² = E[Z_ancilla] in [0, 1]
# - Identical states: overlap² ≈ 1
# - Orthogonal states: overlap² ≈ 0
```

### Coverage Validation

```python
from qamp_shotplanner import coverage_validation_swap, CoverageStats

stats = coverage_validation_swap(
    theta1=0.3,
    theta2=0.8,
    n_trials=100,
    epsilon_F=0.02,
    delta=0.01,
    reference_shots=100000,
    noise_model=None,
)

print(f"Empirical failure rate: {stats.empirical_failure_rate:.2%}")
print(f"Theoretical bound (delta): {stats.delta:.2%}")
print(f"Mean error: {stats.mean_error:.6f}")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_hoeffding.py -v
```

Tests are lightweight and deterministic:
- **test_hoeffding.py**: Formula validation and monotonicity
- **test_swap_identity_statevector.py**: Strong correctness tests using statevector simulation
- **test_swap_planning.py**: SWAP-specific planning matches Hoeffding
- **test_estimator_smoke.py**: Basic EstimatorV2 smoke tests

## Demo Notebook

The `notebooks/01_mvp_swap_hoeffding.ipynb` provides a complete end-to-end demonstration:

1. **State preparation**: Ry(θ1) and Ry(θ2) single-qubit states
2. **Shot planning**: 26,492 shots for εF=0.02, δ=0.01
3. **Three-level decomposition**:
   - Ideal fidelity: F_ideal = cos²((θ1-θ2)/2)
   - Noisy reference: F_device (high-shot reference on device)
   - Hoeffding estimate: F_hat (planned shots)
4. **Coverage validation**: Small-scale validation (100 trials)

Run the notebook top-to-bottom to reproduce the headline results.

## Important Notes

### Estimator Contract

- **Primitive**: Uses Qiskit Aer `EstimatorV2` for expectation value estimation
- **Observable**: Pauli Z on ancilla qubit (index 0) → constructed programmatically with `_z_on_qubit(0, 3)`
  - Qiskit endianness: for 3-qubit system, Z on qubit 0 = `"IIZ"` (rightmost character is qubit 0)
- **Circuit size**: Must be exactly 3 qubits
- **Output**: Returns overlap² = E[Z_ancilla] ∈ [0, 1]
  - Per-shot outcomes: Z_anc ∈ {+1, -1} (bounded in [-1, 1], used for Hoeffding)
  - Expected value: overlap² = E[Z_anc] ∈ [0, 1] (what we return)
  - Identical states: overlap² ≈ 1
  - Orthogonal states: overlap² ≈ 0

### Known Pitfalls

1. **Wrong observable**: The observable construction must account for Qiskit's big-endian Pauli label ordering (rightmost = qubit 0). The helper function `_z_on_qubit()` handles this correctly.
2. **Transpilation**: Circuit is transpiled to ISA before execution (optimization_level=1)
3. **Shot options**: Must specify shots in `run_options`, not `backend_options`
4. **Seed handling**: Use `seed_simulator` in `backend_options` for reproducibility
5. **Range confusion**: Don't mix up per-shot bounds [-1, 1] with expected value bounds [0, 1]

## Project Structure

```
qamp-2025/
├── src/qamp_shotplanner/
│   ├── planners/
│   │   └── hoeffding.py          # HoeffdingPlanner class
│   ├── swaptest/
│   │   ├── circuit.py            # swap_test_1qubit()
│   │   ├── planning.py           # plan_shots_for_swap_fidelity()
│   │   └── run_estimator.py      # run_swap_fidelity_estimator()
│   └── validation/
│       └── coverage.py           # coverage_validation_swap()
├── tests/                        # Lightweight, deterministic tests
├── notebooks/
│   └── 01_mvp_swap_hoeffding.ipynb  # Demo notebook
└── pyproject.toml                # Package configuration
```

## Scope and Limitations

### Week 1-2 Scope (MVP)
✅ Hoeffding planner for bounded variables
✅ SWAP test circuit construction
✅ EstimatorV2 execution wrapper
✅ Coverage validation harness
✅ Demo notebook with reproducible results
✅ Lightweight tests

### Out of Scope (Future Work)
❌ Variance-adaptive planning (Empirical Bernstein)
❌ Anytime stopping
❌ IBM Runtime hardware execution
❌ Error mitigation hooks
❌ Rich CLI or benchmarking frameworks

## Contributing

This is a QAMP 2025 project. See [Issue #53](https://github.com/qiskit-advocate/qamp-2025/issues/53) for project context.

## License

MIT License - See LICENSE file for details.