"""Quickstart: AdaptiveSampler as a certified counterpart to SamplerV2.

SamplerV2 draws a fixed number of shots and returns raw outcomes -- no statistical
claim. A concentration inequality needs a bounded scalar to bind to, so AdaptiveSampler
names one: the probability that a shot lands in a target outcome set. Each shot is the
Bernoulli indicator 1[outcome in S], and the same empirical-Bernstein rule stops as soon
as P(S) is certified to |prob - P(S)| <= eps with probability >= 1 - delta.
"""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qamp_shotplanner import AdaptiveSampler

# A 2-qubit state: 0.8525|00> + 0.5227|11>  (P('11') = sin^2(0.55) ~ 0.273).
qc = QuantumCircuit(2)
qc.ry(1.1, 0)
qc.cx(0, 1)
exact = float(Statevector(qc).probabilities_dict().get("11", 0.0))
print(f"exact P('11') = {exact:.4f}\n")

# Certify the probability of the '11' outcome to +/- eps.
r = AdaptiveSampler(epsilon=0.02, delta=0.01).run([(qc, "11")])[0]
print(f"AdaptiveSampler: P('11') = {r.probability:.4f}   shots = {r.shots:,}   "
      f"certificate: |prob - P| <= {r.epsilon} w.p. >= {1 - r.delta:.2f}")
print(f"  |prob - exact| = {abs(r.probability - exact):.4f}  "
      f"(<= eps: {abs(r.probability - exact) <= r.epsilon})\n")

# Variance-adaptivity: a near-deterministic target costs far fewer shots than p ~ 1/2.
half = QuantumCircuit(1); half.ry(3.14159265 / 2, 0)   # P('1') = 0.50 (max variance)
low = QuantumCircuit(1);  low.ry(0.20, 0)              # P('1') ~ 0.01 (low variance)
s = AdaptiveSampler(epsilon=0.02, delta=0.01)
print(f"P~0.50 target: {s.run([(half, '1')])[0].shots:,} shots")
print(f"P~0.01 target: {s.run([(low, '1')])[0].shots:,} shots   "
      f"(low variance -> stops much earlier)")

# A set of outcomes is a success if a shot matches any of them:
#   r = AdaptiveSampler().run([(qc, ['00', '11'])])[0]   # P('00') + P('11')
# To run on hardware, pass sampler_factory=...; the run() call is unchanged.
