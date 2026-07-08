"""Tests for the drop-in AdaptiveSampler (target-outcome probability)."""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qamp_shotplanner import AdaptiveSampler


def _bell_ish():
    # 0.8525|00> + 0.5227|11>: P('11') = sin^2(0.55) ~ 0.273
    qc = QuantumCircuit(2)
    qc.ry(1.1, 0)
    qc.cx(0, 1)
    return qc


def _exact(qc, bits):
    return float(Statevector(qc).probabilities_dict().get(bits, 0.0))


def test_certifies_target_probability():
    qc = _bell_ish()
    exact = _exact(qc, "11")
    r = AdaptiveSampler(epsilon=0.02, delta=0.01).run([(qc, "11")])[0]
    assert abs(r.probability - exact) <= r.epsilon
    assert r.shots > 0
    assert r.targets == ("11",)


def test_target_set_sums_probabilities():
    qc = _bell_ish()
    # '00' and '11' carry all the amplitude, so P(set) ~ 1.0
    r = AdaptiveSampler(epsilon=0.02, delta=0.01).run([(qc, ["00", "11"])])[0]
    assert abs(r.probability - 1.0) <= r.epsilon


def test_variance_adaptive_low_prob_is_cheaper():
    # p ~ 0.5 (max variance) should cost more shots than a near-deterministic target.
    half = QuantumCircuit(1)
    half.ry(3.14159265 / 2, 0)  # P('1') = 0.5
    low = QuantumCircuit(1)
    low.ry(0.2, 0)  # P('1') = sin^2(0.1) ~ 0.01
    s = AdaptiveSampler(epsilon=0.02, delta=0.01)
    n_half = s.run([(half, "1")])[0].shots
    n_low = s.run([(low, "1")])[0].shots
    assert n_low < n_half


def test_strips_final_measurements():
    qc = _bell_ish()
    qc.measure_all()  # SamplerV2-style measured circuit
    exact = _exact(_bell_ish(), "11")
    r = AdaptiveSampler(epsilon=0.02, delta=0.01).run([(qc, "11")])[0]
    assert abs(r.probability - exact) <= r.epsilon


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        AdaptiveSampler(epsilon=0.0)
    with pytest.raises(ValueError):
        AdaptiveSampler(delta=1.0)
