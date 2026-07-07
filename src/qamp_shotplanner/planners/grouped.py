"""Grouped (commuting-set) shot allocation for multi-Pauli observables.

Estimates ``E = Σ_j c_j ⟨P_j⟩`` (e.g. a Hamiltonian energy) by first partitioning
the Pauli terms into **qubit-wise-commuting (QWC) families**, then running one
Empirical Bernstein stopper *per group* rather than per term. Each family shares a
single tensor-product measurement basis, so the terms inside it are read from ONE
circuit and share shots.

This is the general form of the hand-built 3-group Kagome construction (see
``examples/kagome_heisenberg``): the (eps, delta) guarantee is the thesis's
Bonferroni theorem with the union-bound "items" being **groups** instead of single
Pauli terms. Estimate each group observable ``O_g = Σ_{j∈G_g} c_j P_j`` (range
``R_g = 2 Σ_{j∈G_g}|c_j|``) with the Maurer-Pontil empirical-Bernstein rule at
tolerance ``eps_g`` and budget ``delta_g = delta/G``; if ``Σ_g eps_g ≤ eps`` then
with probability ``≥ 1 - delta``, ``|E_hat − E| ≤ Σ_g eps_g ≤ eps``. See the Kagome
README's "commuting-family EBS guarantee" proposition for the proof.

The cost then scales with the number of *groups* ``G`` and the group variances,
not with ``‖c‖₁²`` — the honest scaling limit of the grouping-free per-term
Bonferroni method (see :mod:`qamp_shotplanner.planners.bonferroni`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from qiskit import QuantumCircuit

from qamp_shotplanner.backends.samplers import SampleMany, ValueMap
from qamp_shotplanner.planners.ebs_stopping import (
    EmpiricalBernsteinStopper,
    StopResult,
)

# A term is a (coefficient, big-endian Pauli label) pair, e.g. (1.0, "IIZZ").
# The label convention matches Qiskit / the Kagome example: label[n-1-q] is the
# single-qubit factor on qubit q, so index 0 is the most-significant qubit.
PauliTerm = tuple[float, str]

# Factory producing a batched sampler for one group's measurement circuit and its
# per-shot value map. The integer is the group index, handy for deterministic
# per-group seeding (e.g. ``lambda qc, vmap, g: statevector_value_sampler(qc, vmap,
# seed=base + g)``).
SamplerFactory = Callable[[QuantumCircuit, ValueMap, int], SampleMany]


@dataclass
class QWCGroup:
    """A qubit-wise-commuting family of Pauli terms sharing one measurement basis.

    Attributes:
        terms: The ``(coeff, label)`` terms assigned to this group.
        basis: Big-endian tensor-product measurement basis, one letter per qubit —
            the non-identity Pauli agreed by every term on that qubit, or ``"I"``
            where all terms act trivially. Same indexing as the term labels
            (``basis[n-1-q]`` is the basis on qubit q).
    """

    terms: list[PauliTerm]
    basis: str

    @property
    def range(self) -> float:
        """Worst-case range ``R_g = 2 Σ_j |c_j|`` of the group observable ``O_g``."""
        return 2.0 * sum(abs(c) for c, _ in self.terms)


@dataclass
class GroupedResult:
    """Result of a grouped (commuting-set) multi-term energy estimate.

    Attributes:
        energy: The combined estimate ``E_hat = Σ_g μ̂_g``.
        total_shots: Total shots used across all groups.
        certified_eps: The certified error bound ``Σ_g eps_g`` (``≤ eps``), valid
            with probability ``≥ 1 - delta`` by the union bound over groups.
        groups: The QWC groups, ordered as produced by the grouping front-end.
        eps_alloc: Per-group tolerance ``eps_g`` actually used (parallel to
            ``groups`` and ``per_group``).
        per_group: Per-group :class:`StopResult`, ordered as ``groups``.
    """

    energy: float
    total_shots: int
    certified_eps: float
    groups: list[QWCGroup]
    eps_alloc: list[float]
    per_group: list[StopResult]


def _qwc_compatible(basis: str, label: str) -> bool:
    """True if ``label`` can be measured in ``basis`` (qubit-wise commuting).

    Two Paulis are QWC iff on every qubit their single-qubit factors are equal or
    at least one is the identity. Here ``basis`` is the running measurement basis of
    a group (identity where still unconstrained); a term joins iff each of its
    non-identity factors matches the group's letter on that qubit or the group is
    still unconstrained (``"I"``) there.
    """
    for b, p in zip(basis, label):
        if p != "I" and b != "I" and p != b:
            return False
    return True


def _merge_basis(basis: str, label: str) -> str:
    """Fill the group's identity slots with ``label``'s non-identity factors."""
    return "".join(p if b == "I" else b for b, p in zip(basis, label))


def qubitwise_commuting_groups(terms: Sequence[PauliTerm]) -> list[QWCGroup]:
    """Greedily partition Pauli terms into qubit-wise-commuting (QWC) groups.

    Two Paulis are QWC iff on every qubit their single-qubit factors are equal or
    one is the identity — exactly the terms that can be read simultaneously from a
    single tensor-product measurement basis (single-qubit rotations only). Uses
    first-fit greedy assignment: each term joins the first existing group it is QWC
    with (updating that group's basis), else it opens a new group. This is a
    standard, cheap heuristic; it does not minimise the group count (that is
    graph-colouring on the commutation graph), but it is deterministic and order-
    stable, which keeps seeding and reporting reproducible.

    Args:
        terms: ``(coeff, label)`` pairs. All labels must share the same length
            (number of qubits), big-endian as in Qiskit Pauli strings.

    Returns:
        A list of :class:`QWCGroup`, each carrying its terms and the merged
        tensor-product measurement basis.

    Raises:
        ValueError: If ``terms`` is empty or the labels have differing lengths.
    """
    if not terms:
        raise ValueError("terms must be non-empty")
    n = len(terms[0][1])
    if any(len(label) != n for _, label in terms):
        raise ValueError("all Pauli labels must have the same length")

    groups: list[QWCGroup] = []
    for coeff, label in terms:
        for group in groups:
            if _qwc_compatible(group.basis, label):
                group.terms.append((coeff, label))
                group.basis = _merge_basis(group.basis, label)
                break
        else:
            groups.append(QWCGroup(terms=[(coeff, label)], basis=label))
    return groups


def _rotate_into_basis(state_circuit: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Copy ``state_circuit`` and rotate every qubit into its group basis.

    ``basis`` is big-endian (``basis[n-1-q]`` is the basis on qubit q). X → H,
    Y → Sdg·H, Z/I → no rotation, so a subsequent Z-basis measurement reads the
    Pauli eigenvalue on each qubit.
    """
    n = state_circuit.num_qubits
    qc = state_circuit.copy()
    for q in range(n):
        letter = basis[n - 1 - q]
        if letter == "X":
            qc.h(q)
        elif letter == "Y":
            qc.sdg(q)
            qc.h(q)
    return qc


def _group_value_map(group: QWCGroup, n: int) -> ValueMap:
    """Per-shot value map ``Σ_j c_j · eigenvalue(P_j)`` for a rotated group circuit.

    After rotating into the group basis and measuring in Z, the eigenvalue of a
    term ``P_j`` on a shot is ``(-1)^(parity of the measured bits under P_j's
    support)``. Precomputes each term's support (qubit indices) once.
    """
    supports = [
        (coeff, [q for q in range(n) if label[n - 1 - q] != "I"])
        for coeff, label in group.terms
    ]

    def value(bitstring: str) -> float:
        bits = bitstring.replace(" ", "")
        total = 0.0
        for coeff, support in supports:
            parity = 0
            for q in support:
                parity ^= int(bits[len(bits) - 1 - q])
            total += coeff * (1.0 if parity == 0 else -1.0)
        return total

    return value


def _allocate_eps(groups: list[QWCGroup], eps: float, split: str) -> list[float]:
    """Split the accuracy budget ``eps`` over groups so that ``Σ_g eps_g = eps``.

    Because ``E = Σ_g ⟨O_g⟩`` combines groups with unit weight, the certified error
    is ``Σ_g eps_g`` (no ‖c‖₁ factor at the group level). Two splits, both keeping
    ``Σ_g eps_g = eps`` so the guarantee holds unchanged:

    - ``"uniform"``: ``eps_g = eps / G`` for every group.
    - ``"tight"``: ``eps_g ∝ R_g^{2/3}``. Minimising the Hoeffding leading cost
      ``Σ_g R_g² / eps_g²`` subject to ``Σ_g eps_g = eps`` gives, by a Lagrange
      argument, ``eps_g ∝ R_g^{2/3}``. Range is a worst-case proxy for variance
      (``Var(O_g) ≤ (R_g/2)²``) and is known a priori, so this needs no pilot pass;
      a true variance-aware split ``eps_g ∝ Var(O_g)^{1/3}`` (mirroring the per-term
      Bonferroni allocation) would require estimating the group variances first and
      is left to the caller.
    """
    G = len(groups)
    if split == "uniform":
        return [eps / G] * G
    if split == "tight":
        raw = [g.range ** (2.0 / 3.0) for g in groups]
        denom = sum(raw)
        if denom == 0:
            return [eps / G] * G
        return [eps * r / denom for r in raw]
    raise ValueError(f"unknown split {split!r}; use 'uniform' or 'tight'")


def grouped_energy_estimate(
    terms: Sequence[PauliTerm],
    state_circuit: QuantumCircuit,
    eps: float,
    delta: float,
    sampler_factory: SamplerFactory,
    split: Literal["uniform", "tight"] = "uniform",
) -> GroupedResult:
    """Estimate ``E = Σ_j c_j ⟨P_j⟩`` via qubit-wise-commuting group estimation.

    Partitions ``terms`` into QWC groups (:func:`qubitwise_commuting_groups`); for
    each group builds one measurement circuit (rotate every qubit into the group
    basis) and a per-shot value map ``Σ_j c_j · eigenvalue(P_j)``; wraps it as a
    bounded observable of range ``R_g = 2 Σ_j |c_j|`` and runs an
    :class:`EmpiricalBernsteinStopper` at budget ``delta_g = delta / G`` and
    tolerance ``eps_g``. The group estimates combine by the union bound:
    ``E_hat = Σ_g μ̂_g`` is ``(Σ_g eps_g, delta)``-correct, and ``Σ_g eps_g ≤ eps``
    by construction of the split.

    Args:
        terms: ``(coeff, label)`` Pauli terms of the observable, big-endian labels.
        state_circuit: The (unmeasured) state-preparation circuit to estimate on.
        eps: Target accuracy on the combined energy.
        delta: Total failure probability, in ``(0, 1)``.
        sampler_factory: ``(circuit, value_map, group_index) -> SampleMany``; builds
            the batched sampler for a group (e.g. wrapping
            :func:`statevector_value_sampler` with a per-group seed).
        split: ``"uniform"`` (``eps_g = eps/G``) or ``"tight"``
            (``eps_g ∝ R_g^{2/3}``). Both certify ``Σ_g eps_g = eps``.

    Returns:
        A :class:`GroupedResult` with the combined energy, total shots, per-group
        stopper results, the per-group tolerances used, and the certified bound
        ``Σ_g eps_g``.
    """
    n = state_circuit.num_qubits
    groups = qubitwise_commuting_groups(terms)
    eps_alloc = _allocate_eps(groups, eps, split)
    delta_g = delta / len(groups)

    energy = 0.0
    total_shots = 0
    per_group: list[StopResult] = []

    for g, (group, eps_g) in enumerate(zip(groups, eps_alloc)):
        half = group.range / 2.0
        circuit = _rotate_into_basis(state_circuit, group.basis)
        value_map = _group_value_map(group, n)
        sampler = sampler_factory(circuit, value_map, g)
        stopper = EmpiricalBernsteinStopper(
            epsilon_stat=eps_g,
            delta=delta_g,
            a=-half,
            b=half,
        )
        result = stopper.run_batched(sampler)
        per_group.append(result)
        energy += result.estimate
        total_shots += result.n

    return GroupedResult(
        energy=energy,
        total_shots=total_shots,
        certified_eps=sum(eps_alloc),
        groups=groups,
        eps_alloc=eps_alloc,
        per_group=per_group,
    )
