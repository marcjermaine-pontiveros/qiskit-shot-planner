"""Noise models for the Aer-backed samplers (harvested from PCSC/QCE experiments).

Three setups are exposed, matching the thesis Experimental Setup:
    - fake_montreal_simulator: calibration-seeded FakeMontrealV2 backend (exp3).
    - depolarizing_noise_model: parametric 1q/2q depolarizing (exp2/exp4).
    - calibrated_noise_model: depolarizing + readout error sweep (exp_qce2).

`qiskit_aer` and `qiskit_ibm_runtime` are imported lazily so importing this module
never hard-fails when the Aer/runtime extras are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel


def fake_montreal_simulator() -> "AerSimulator":
    """AerSimulator seeded from FakeMontrealV2 (real IBM Montreal calibration).

    Wraps ``AerSimulator.from_backend(FakeMontrealV2())`` so the simulator inherits
    the recorded calibration of the 27-qubit IBM Montreal device: per-qubit T1/T2,
    per-gate error rates, and per-qubit readout error. No parametric channel is
    added by hand; the noise is whatever the snapshot encodes.

    Used by exp3 (FakeMontrealV2 validation) as the realistic development/validation
    backend at eps=0.02, delta=0.01.

    Returns:
        AerSimulator configured from the FakeMontrealV2 calibration snapshot.

    Raises:
        ImportError: if ``qiskit_aer`` or ``qiskit_ibm_runtime`` is unavailable.
    """
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMontrealV2

    return AerSimulator.from_backend(FakeMontrealV2())


def depolarizing_noise_model(p1: float, p2: float) -> "NoiseModel":
    """Parametric depolarizing noise on 1q and 2q gates (exp2/exp4).

    Channels (via ``depolarizing_error``, all-qubit):
        - 1q depolarizing with parameter ``p1`` on ["h", "rx"].
        - 2q depolarizing with parameter ``p2`` on ["cx"].

    exp4 instantiates this as ``p1 = p/10`` on single-qubit gates and ``p2 = p`` on
    the CX, with the base noise level ``p = 0.05`` (so p1 = 0.005, p2 = 0.05). exp2
    sweeps the effective noise across p in {0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10,
    0.20, 0.30, 0.50}.

    Args:
        p1: Single-qubit depolarizing parameter (applied to h, rx).
        p2: Two-qubit depolarizing parameter (applied to cx).

    Returns:
        NoiseModel with the two all-qubit depolarizing channels.

    Raises:
        ImportError: if ``qiskit_aer`` is unavailable.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "rx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def calibrated_noise_model(p: float, readout: float) -> "NoiseModel":
    """Calibrated depolarizing + readout noise model (exp_qce2).

    Channels:
        - 1q depolarizing with parameter ``p`` on ["h", "rx", "ry", "rz", "x"].
        - 2q depolarizing with parameter ``min(5 * p, 1.0)`` on ["cx"] (two-qubit
          gates are ~5x noisier than single-qubit gates).
        - Symmetric readout error ``[[1-readout, readout], [readout, 1-readout]]``
          applied to all qubits.

    exp_qce2 ties readout to the depolarizing level as ``readout = min(0.5 * p, 0.5)``
    and sweeps p in {0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10}; at p = 0 the model is
    the identity (noiseless). Here ``readout`` is exposed as an explicit parameter.

    Args:
        p: Single-qubit depolarizing parameter; the CX rate is min(5*p, 1.0).
        readout: Symmetric bit-flip probability for the readout error.

    Returns:
        NoiseModel with depolarizing channels plus all-qubit readout error.

    Raises:
        ImportError: if ``qiskit_aer`` is unavailable.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    if p > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(p, 1), ["h", "rx", "ry", "rz", "x"]
        )
        nm.add_all_qubit_quantum_error(
            depolarizing_error(min(p * 5, 1.0), 2), ["cx"]
        )
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
        )
    return nm
