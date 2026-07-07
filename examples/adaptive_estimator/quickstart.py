"""Quickstart: AdaptiveEstimator as a drop-in for EstimatorV2, but with a certificate.

Same `run([(circuit, observable)])` shape as Qiskit's EstimatorV2 — except instead of
a fixed target precision with no coverage guarantee, you get a value certified to
|value - <O>| <= eps with probability >= 1-delta, plus the shots that cost.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector

from qamp_shotplanner import AdaptiveEstimator

# A small workload: a 2-qubit state and a 3-term observable.
qc = QuantumCircuit(2)
qc.ry(1.1, 0)
qc.cx(0, 1)
obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("II", -0.3)])
exact = float(Statevector(qc).expectation_value(obs).real)
print(f"exact <O> = {exact:.4f}\n")

# --- The usual EstimatorV2 way: a fixed precision, no coverage guarantee ---
ev2 = StatevectorEstimator(default_precision=0.02)
v = float(ev2.run([(qc, obs)]).result()[0].data.evs)
print(f"EstimatorV2 (precision=0.02): value = {v:+.4f}   "
      f"(a 1-sigma standard-error target; NOT a coverage guarantee)")

# --- The drop-in adaptive way: same call, but certified ---
r = AdaptiveEstimator(epsilon=0.02, delta=0.01).run([(qc, obs)])[0]
print(f"AdaptiveEstimator          : value = {r.value:+.4f}   "
      f"shots = {r.shots:,}   certificate: |value - <O>| <= {r.epsilon} w.p. >= {1-r.delta:.2f}")
print(f"  |value - exact| = {abs(r.value - exact):.4f}  (<= eps: {abs(r.value-exact) <= r.epsilon})")

# Swap a hardware/noisy sampler in without changing the call:
#   est = AdaptiveEstimator(sampler_factory=lambda c, m, s: backend_sampler(c, m, backend))
#   est.run([(circuit, observable)])
print("\nTo run on hardware, pass sampler_factory=... (backend_sampler); the run() call is unchanged.")
