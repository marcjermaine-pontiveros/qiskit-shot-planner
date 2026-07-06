# Budgeting circuit runs for expectation values using concentration bounds

## Overview

`qiskit-shot-planner` (package `qamp_shotplanner`) plans and adaptively stops the
number of shots needed to estimate quantum expectation values with explicit
finite-sample `(ε, δ)` guarantees. It is the reference implementation for the
MS thesis *Adaptive Shot Allocation for Quantum Observable Estimation via
Concentration Inequalities*, grown from the QAMP 2025 prototype.

## Features (v0.2.0)

- **Hoeffding planner** — fixed-budget shot planning for bounded variables `X ∈ [a, b]`.
- **Empirical Bernstein stopping** — variance-adaptive early stopping on the provable
  two-sided **Maurer–Pontil** radius, via a geometric-checkpoint rule.
- **Anytime-valid stopping** — a confidence-sequence variant valid at every sample count.
- **Bonferroni multi-Pauli** — joint `(ε, δ)` energy guarantee across Hamiltonian terms,
  with uniform / tight / variance-aware allocation.
- **Le Cam lower bounds** — two-point sample-complexity floors for the optimality benchmark.
- **Backends** — sampler adapters, single-knob depolarizing+readout noise models, and
  IBM Runtime run provenance (job capture + manifest).
- **Error mitigation** — ZNE gate folding with a variance-inflation (`γ²`) diagnostic.
- **Workloads** — SWAP-test fidelity, QAOA MaxCut, and five-term H₂ VQE.

The stopping rules control sampling error relative to the (possibly biased) device-level
expectation; hardware bias is kept explicitly outside the guarantee via a three-level
error hierarchy. To reproduce the thesis numbers, check out the `v0.2.0` tag.

## Installation

```bash
# Install in editable mode
uv pip install -e .

# Or with pip
pip install -e .
```

## Using with Your Own Circuits

This library works with any Qiskit circuit and any bounded observable. Here's the basic pattern:

```python
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2
from qamp_shotplanner import HoeffdingPlanner, pauli_z

# 1. Create your circuit
qc = QuantumCircuit(1)
qc.ry(0.5, 0)

# 2. Define observable (Pauli observables bounded in [-1, 1])
observable = pauli_z(qubit=0, num_qubits=1)

# 3. Plan shots with Hoeffding bound
planner = HoeffdingPlanner(
    epsilon_stat=0.02,  # Error tolerance
    delta=0.01,         # Failure probability
    a=-1.0,             # Observable lower bound
    b=1.0               # Observable upper bound
)
shots = planner.planned_shots()  # 26,492 shots

# 4. Run with Qiskit EstimatorV2
estimator = EstimatorV2(options={"run_options": {"shots": shots}})
job = estimator.run([(qc, observable)])
result = job.result()[0]
expectation_value = float(result.data.evs)
```

This works for:
- **Single-qubit Paulis**: Use `pauli_x()`, `pauli_y()`, `pauli_z()` helpers
- **Multi-qubit correlations**: Use `correlation_observable()` for ZZ, XX, etc.
- **Hamiltonians**: Use `hamiltonian_term()` for energy estimation

See the demo notebooks for more examples of multi-qubit and Hamiltonian use cases.

## Using with SamplerV2

The Sampler primitive returns quasi-probability distributions instead of expectation values.
You can still use HoeffdingPlanner - just compute the expectation value from the sampled counts.

### Local BackendSamplerV2

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2
from qamp_shotplanner import HoeffdingPlanner

# 1. Create circuit with measurement
qc = QuantumCircuit(1)
qc.ry(0.5, 0)
qc.measure_all()  # Required for Sampler

# 2. Plan shots for Z observable (bounded in [-1, 1])
planner = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
shots = planner.planned_shots()

# 3. Run with BackendSamplerV2
backend = AerSimulator()
sampler = BackendSamplerV2(backend=backend)
job = sampler.run([qc], shots=shots)
result = job.result()[0]

# 4. Post-process to get expectation value
# For Z measurement: E[Z] = P(0) - P(1)
quasi_dist = result.data.meas
expectation_value = quasi_dist.get(0, 0) - quasi_dist.get(1, 0)
```

### IBM Runtime SamplerV2

```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_pass_managers import generate_preset_pass_manager
from qamp_shotplanner import HoeffdingPlanner

# 1. Initialize and select backend
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# 2. Create and transpile circuit to ISA
qc = QuantumCircuit(1)
qc.ry(0.5, 0)
qc.measure_all()

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# 3. Plan shots (same as local)
planner = HoeffdingPlanner(epsilon_stat=0.02, delta=0.01, a=-1.0, b=1.0)
shots = planner.planned_shots()

# 4. Run with IBM Runtime Sampler
sampler = SamplerV2(mode=backend)
job = sampler.run([isa_circuit], shots=shots)
result = job.result()[0]

# 5. Post-process quasi-distribution into an expectation value
quasi_dist = result.data.c
expectation_value = quasi_dist.get(0, 0) - quasi_dist.get(1, 0)
```

**Note**: For observables other than Pauli-Z, you need to rotate the measurement basis:
- For X: add `qc.h(qubit)` before measuring
- For Y: add `qc.sdg(qubit); qc.h(qubit)` before measuring

## SWAP Test Example

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

### Observable Helpers

```python
from qamp_shotplanner import (
    # Single-qubit observables
    pauli_x,
    pauli_y,
    pauli_z,
    single_qubit_observable,

    # Multi-qubit correlations
    correlation_observable,
    bell_state_observable,

    # Hamiltonian terms
    hamiltonian_term,
)

# Single-qubit Z on qubit 0 in 2-qubit system
obs_z = pauli_z(qubit=0, num_qubits=2)  # Returns SparsePauliOp "IZ"

# Multi-qubit ZZ correlation
obs_zz = correlation_observable(qubit1=0, qubit2=1, num_qubits=2, pauli1="Z", pauli2="Z")

# Bell state observable helper
obs_xx = bell_state_observable(num_qubits=2, correlation_type="XX")

# Hamiltonian term: -0.5 * Z_0 * Z_1
term = hamiltonian_term(qubits=(0, 1), paulis=("Z", "Z"), coefficient=-0.5, num_qubits=2)
```

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
- **test_observables.py**: Observable helper functions (21 tests)

## Demo Notebooks

The package includes four reproducible notebooks demonstrating different quantum estimation tasks:

### 1. SWAP Test Fidelity (`01_mvp_swap_hoeffding.ipynb`)
Complete end-to-end demonstration:
- **State preparation**: Ry(θ1) and Ry(θ2) single-qubit states
- **Shot planning**: 26,492 shots for εF=0.02, δ=0.01
- **Three-level decomposition**:
  - Ideal fidelity: F_ideal = cos²((θ1-θ2)/2)
  - Noisy reference: F_device (high-shot reference on device)
  - Hoeffding estimate: F_hat (planned shots)
- **Coverage validation**: Small-scale validation (100 trials)

### 2. Observables Basics (`02_observables_basics.ipynb`)
Demonstrates that Hoeffding planning works for any bounded observable:
- **Single-qubit Pauli observables**: X, Y, Z measurements
- **Universal shot planning**: Same planning for all Pauli operators (eigenvalues ±1)
- **Theoretical validation**: Compare measured vs. predicted expectation values
- **Multiple states**: Explore how observables change with different rotation angles

### 3. Multi-Qubit Observables (`03_multi_qubit_observables.ipynb`)
Explores quantum correlations and entanglement:
- **Correlation observables**: ZZ, XX, YY measurements on two-qubit systems
- **Product vs. entangled states**: Compare |0⟩⊗|+⟩ vs. Bell state |Φ⁺⟩
- **Bell's theorem**: Observe correlations impossible in classical systems
- **Mixed correlations**: XZ, ZY and other multi-qubit observables

### 4. Hamiltonian Expectation (`04_hamiltonian_expectation.ipynb`)
Practical quantum simulation example:
- **Transverse Ising model**: Real physics Hamiltonian decomposition
- **Term-wise estimation**: Plan different shots for each Hamiltonian term
- **Error allocation**: Distribute error budget across terms
- **Variational optimization preview**: Energy minimization workflow

Run each notebook top-to-bottom to reproduce the results.

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
│   ├── observables/
│   │   ├── basics.py             # Single-qubit observable helpers
│   │   └── multi_qubit.py        # Multi-qubit correlation & Hamiltonian helpers
│   └── validation/
│       └── coverage.py           # coverage_validation_swap()
├── tests/                        # Lightweight, deterministic tests
│   └── test_observables.py       # Observable helper tests (21 tests)
├── notebooks/
│   ├── 01_mvp_swap_hoeffding.ipynb      # SWAP test demo
│   ├── 02_observables_basics.ipynb      # Single-qubit observables
│   ├── 03_multi_qubit_observables.ipynb # Multi-qubit correlations
│   └── 04_hamiltonian_expectation.ipynb # Hamiltonian estimation
└── pyproject.toml                # Package configuration
```

## Scope and Limitations

### In Scope (v0.2.0)
✅ Hoeffding, Empirical Bernstein (Maurer–Pontil), geometric-checkpoint, and anytime stopping
✅ Bonferroni multi-Pauli energy guarantee with variance-aware allocation
✅ Le Cam two-point lower bounds
✅ Noise models, IBM Runtime provenance, and ZNE with a variance-inflation diagnostic
✅ SWAP-test, QAOA, and H₂ VQE workloads
✅ Coverage validation harness and a full test suite

### Out of Scope (Future Work)
❌ Live session-based hardware stopping (open plan runs the resumable job-mode driver instead)
❌ PEC and other mitigation families beyond ZNE
❌ Rich CLI or general benchmarking frameworks

## Reproducibility

`scripts/` regenerates the thesis manuscript artifacts (tables, figures, hardware, online runs);
`examples/pcsc2026/` holds the PCSC/QCE paper experiment drivers. Both run on this library.

## License

MIT License — see the [LICENSE](LICENSE) file.