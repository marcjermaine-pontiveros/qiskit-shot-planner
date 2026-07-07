"""Tests for the general multi-valued statevector sampler."""
import numpy as np
from qiskit import QuantumCircuit

from qamp_shotplanner import statevector_value_sampler


def test_bell_parity_is_deterministic():
    """A Bell state read in the Z basis has fixed even parity -> value +1 always."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    def parity(bits: str) -> float:
        b = bits.replace(" ", "")
        return (-1.0) ** (int(b[0]) ^ int(b[1]))

    sampler = statevector_value_sampler(qc, parity, seed=0)
    samples = sampler(500)
    assert len(samples) == 500
    assert all(v == 1.0 for v in samples)  # |00> and |11> both have even parity


def test_multivalued_mean_matches_expectation():
    """Sample mean of a 3-level observable converges to its statevector expectation."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)  # uniform over {00,01,10,11}

    def count_ones(bits: str) -> float:
        return float(bits.replace(" ", "").count("1"))  # takes values 0,1,2

    sampler = statevector_value_sampler(qc, count_ones, seed=1)
    mean = float(np.mean(sampler(20000)))
    assert abs(mean - 1.0) < 0.05  # E[#ones] = 1.0 for two uniform bits


def test_seed_is_reproducible():
    qc = QuantumCircuit(1)
    qc.h(0)
    vmap = lambda bits: float(bits.replace(" ", "").count("1"))
    a = statevector_value_sampler(qc, vmap, seed=42)(100)
    b = statevector_value_sampler(qc, vmap, seed=42)(100)
    assert a == b
