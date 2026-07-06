"""Zero-noise extrapolation via global gate folding.

Harvested from PCSC 2026 exp4 (``fold_gates``, ``zne_linear``,
``zne_exponential``). Noise is amplified by folding each gate G → G (G† G)^(k-1),
then the observable is Richardson-extrapolated back to the zero-noise limit.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit


def fold_gates(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """Amplify noise by global gate folding.

    Replaces each gate G with G (G† G)^(factor-1), leaving the ideal unitary
    unchanged while scaling the effective noise by ``factor``.

    Args:
        circuit: Circuit to fold.
        factor: Integer noise scale factor (>= 1). ``factor=1`` returns a copy.

    Returns:
        The folded circuit.

    Raises:
        ValueError: If ``factor`` < 1.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return circuit.copy()

    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instruction in circuit.data:
        gate = instruction.operation
        qargs = instruction.qubits
        folded.append(gate, qargs)
        for _ in range(factor - 1):
            folded.append(gate.inverse(), qargs)
            folded.append(gate, qargs)
    return folded


def _extrapolate_linear(factors: Sequence[float], values: Sequence[float]) -> float:
    coeffs = np.polyfit(factors, values, deg=1)
    return float(np.polyval(coeffs, 0))


def _extrapolate_exponential(
    factors: Sequence[float], values: Sequence[float]
) -> float:
    def model(f, A, b, C):
        return A * np.exp(-b * np.array(f)) + C

    try:
        popt, _ = curve_fit(
            model, factors, values, p0=[0.5, 0.1, 0.5], maxfev=5000
        )
        return float(model(0, *popt))
    except Exception:
        return _extrapolate_linear(factors, values)


def zne_extrapolate(
    factors: Sequence[float],
    values: Sequence[float],
    method: Literal["linear", "exponential"] = "linear",
) -> float:
    """Richardson-extrapolate observable values to the zero-noise limit.

    Args:
        factors: Noise scale factors (x-axis).
        values: Observable estimate at each factor (y-axis).
        method: "linear" (degree-1 polyfit) or "exponential"
            (A·exp(-b·f) + C, falling back to linear if the fit fails).

    Returns:
        The extrapolated value at factor = 0.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "linear":
        return _extrapolate_linear(factors, values)
    if method == "exponential":
        return _extrapolate_exponential(factors, values)
    raise ValueError(f"unknown method: {method!r}")
