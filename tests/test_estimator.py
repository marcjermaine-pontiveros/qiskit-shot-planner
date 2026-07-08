"""Tests for the drop-in AdaptiveEstimator."""
import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qamp_shotplanner import AdaptiveEstimator


def _state():
    qc = QuantumCircuit(2)
    qc.ry(1.1, 0)
    qc.cx(0, 1)
    return qc


def test_estimate_within_eps_of_exact():
    qc = _state()
    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("II", -0.3)])
    exact = float(Statevector(qc).expectation_value(obs).real)
    r = AdaptiveEstimator(epsilon=0.02, delta=0.01).run([(qc, obs)])[0]
    assert abs(r.value - exact) <= r.epsilon
    assert r.shots > 0
    assert r.n_terms == 2  # ZZ, XI (II is exact, not sampled)


def test_pure_constant_observable_costs_no_shots():
    qc = _state()
    obs = SparsePauliOp.from_list([("II", 1.7)])
    r = AdaptiveEstimator().run([(qc, obs)])[0]
    assert r.value == 1.7
    assert r.shots == 0
    assert r.n_terms == 0


def test_multiple_pubs_indexable():
    qc = _state()
    obs0 = SparsePauliOp.from_list([("ZI", 1.0)])
    obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
    res = AdaptiveEstimator(epsilon=0.03, delta=0.01).run([(qc, obs0), (qc, obs1)])
    assert len(res) == 2
    e0 = float(Statevector(qc).expectation_value(obs0).real)
    e1 = float(Statevector(qc).expectation_value(obs1).real)
    assert abs(res[0].value - e0) <= 0.03
    assert abs(res[1].value - e1) <= 0.03


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        AdaptiveEstimator(epsilon=0.0)
    with pytest.raises(ValueError):
        AdaptiveEstimator(delta=1.0)
