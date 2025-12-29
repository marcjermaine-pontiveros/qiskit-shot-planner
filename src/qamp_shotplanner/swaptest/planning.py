"""Shot planning for SWAP test fidelity estimation."""

from qamp_shotplanner.planners.hoeffding import HoeffdingPlanner


def plan_shots_for_swap_fidelity(epsilon_F: float, delta: float) -> int:
    """Plan shots for SWAP test fidelity estimation using Hoeffding's bound.

    Plans the number of shots required so that the SWAP-test overlap² estimate
    F_hat satisfies |F_hat - F| <= epsilon_F with failure probability <= delta.

    The SWAP test estimates overlap-squared: F = E[Z_ancilla] where Z_anc ∈ {+1, -1}.
    Hoeffding's bound is applied to the per-shot variable Z_anc (bounded in [-1, 1]),
    which guarantees the bound on the expected value F ∈ [0, 1].

    Args:
        epsilon_F: Tolerance on overlap² estimate error |F_hat - F|
        delta: Failure probability (must be in (0, 1))

    Returns:
        Number of shots required

    Raises:
        ValueError: If epsilon_F <= 0 or delta not in (0, 1)
    """
    planner = HoeffdingPlanner(
        epsilon_stat=epsilon_F,
        delta=delta,
        a=-1.0,  # Per-shot Z_anc ∈ [-1, 1]
        b=+1.0,
    )
    return planner.planned_shots()