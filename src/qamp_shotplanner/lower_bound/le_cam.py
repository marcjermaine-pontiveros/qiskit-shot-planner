"""Le Cam lower bound on expected shots for Pauli observables (thesis Appendix B)."""

from __future__ import annotations

from math import log, sqrt


def le_cam_lower_bound(eps: float, delta: float, sigma2: float, R: float = 2.0) -> float:
    """Le Cam lower bound on E[τ] for any (ε, δ)-correct sequential procedure.

    For a Pauli observable (R = 2) of variance σ² and δ ≤ 1/4,
        E[τ] ≥ (1/12) · max(σ²·log(1/δ)/ε², R·log(1/δ)/ε).
    The first term is the variance (sub-Gaussian) regime, the second the range
    (worst-case) regime. Derived via a two-point Bernoulli reduction with
    D_KL ≤ χ² = 4ε²/σ² and the Kaufmann–Cappé–Garivier transportation lemma;
    the δ ≤ 1/4 requirement enters through (1−2δ)log((1−δ)/δ) ≥ (1/3)log(1/δ).

    Args:
        eps: Estimation tolerance ε > 0.
        delta: Failure probability, must satisfy 0 < δ ≤ 1/4.
        sigma2: Observable variance σ² ≥ 0.
        R: Observable range (2 for a Pauli with ±1 eigenvalues), must be > 0.

    Returns:
        Lower bound on the expected number of shots E[τ].

    Raises:
        ValueError: If eps <= 0, delta not in (0, 1/4], sigma2 < 0, or R <= 0.
    """
    _validate(eps, delta, sigma2, R)
    log_inv_delta = log(1.0 / delta)
    variance_term = sigma2 * log_inv_delta / eps**2
    range_term = R * log_inv_delta / eps
    return (1.0 / 12.0) * max(variance_term, range_term)


def le_cam_additive(eps: float, delta: float, sigma2: float, R: float = 2.0) -> float:
    """Additive form of the Le Cam bound with c₁ = c₂ = 1/24.

    E[τ] ≥ (1/24)·(σ²·log(1/δ)/ε² + R·log(1/δ)/ε). Sums both regimes instead of
    taking the max; it is weaker by at most a factor of two but smooth in σ².

    Args:
        eps: Estimation tolerance ε > 0.
        delta: Failure probability, must satisfy 0 < δ ≤ 1/4.
        sigma2: Observable variance σ² ≥ 0.
        R: Observable range, must be > 0.

    Returns:
        Additive lower bound on the expected number of shots E[τ].

    Raises:
        ValueError: If eps <= 0, delta not in (0, 1/4], sigma2 < 0, or R <= 0.
    """
    _validate(eps, delta, sigma2, R)
    log_inv_delta = log(1.0 / delta)
    variance_term = sigma2 * log_inv_delta / eps**2
    range_term = R * log_inv_delta / eps
    return (1.0 / 24.0) * (variance_term + range_term)


def _bernoulli_kl(p: float, q: float) -> float:
    """KL divergence D_KL(Bern(p) ‖ Bern(q)) in nats, clipped for safety."""
    c = 1e-12
    p = min(max(p, c), 1.0 - c)
    q = min(max(q, c), 1.0 - c)
    return p * log(p / q) + (1.0 - p) * log((1.0 - p) / (1.0 - q))


def le_cam_two_point(eps: float, delta: float, sigma2: float, R: float = 2.0) -> float:
    """Tight two-point Le Cam lower bound on E[τ] for a Pauli observable.

    Sharper than :func:`le_cam_lower_bound`: it uses the exact Bernoulli KL of the
    two-point family (rather than the loose χ² surrogate) and the exact
    Kaufmann–Cappé–Garivier factor d(1−δ‖δ) (rather than the (1/3)log(1/δ) bound).

    A Pauli measurement is a ±1 Bernoulli with p = (1+μ)/2 and σ² = 1−μ². Two
    hypotheses whose means differ by 2ε differ by ε in p. For a straddling pair
    (p₀, p₁) with p₁−p₀ = ε, the transportation lemma gives
        E[τ] ≥ d(1−δ‖δ) / min(D_KL(P₀‖P₁), D_KL(P₁‖P₀)),  d(1−δ‖δ) = (1−2δ)log((1−δ)/δ).
    The bound is the largest such value over the feasible placements at variance σ².

    Args:
        eps: Estimation tolerance ε > 0.
        delta: Failure probability, 0 < δ ≤ 1/4.
        sigma2: Observable variance σ² ∈ [0, 1] for a Pauli.
        R: Observable range (unused here beyond validation; kept for signature parity).

    Returns:
        Tight lower bound on the expected number of shots E[τ].

    Raises:
        ValueError: If eps <= 0, delta not in (0, 1/4], sigma2 < 0, or R <= 0.
    """
    _validate(eps, delta, sigma2, R)
    mu = sqrt(max(0.0, 1.0 - sigma2))  # |μ|; the bound is symmetric in sign(μ)
    p = 0.5 * (1.0 + mu)
    dp = eps  # a 2ε separation in mean is an ε separation in p
    d = (1.0 - 2.0 * delta) * log((1.0 - delta) / delta)

    best = 0.0
    for lo in (p - dp, p):  # the two straddling placements around p
        hi = lo + dp
        if lo <= 0.0 or hi >= 1.0:
            continue
        kl = min(_bernoulli_kl(lo, hi), _bernoulli_kl(hi, lo))
        if kl > 0.0:
            best = max(best, d / kl)
    if best == 0.0:  # near the boundary (σ² → 0): clip a feasible pair inside (0, 1)
        hi = min(p, 1.0 - 1e-9)
        lo = max(hi - dp, 1e-9)
        kl = min(_bernoulli_kl(lo, hi), _bernoulli_kl(hi, lo))
        best = d / kl if kl > 0.0 else 0.0
    return best


def _validate(eps: float, delta: float, sigma2: float, R: float) -> None:
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < delta <= 0.25):
        raise ValueError("delta must be in (0, 1/4]")
    if sigma2 < 0:
        raise ValueError("sigma2 must be >= 0")
    if R <= 0:
        raise ValueError("R must be > 0")
