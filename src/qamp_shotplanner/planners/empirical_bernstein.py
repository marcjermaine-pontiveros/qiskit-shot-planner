"""Empirical Bernstein inequality functions for adaptive stopping.

Based on Algorithm 2 and Eq. (4) from:
Gresch, Tepe, and Kliesch, "Adaptive and provably accurate estimation of
quantum expectation values using the empirical Bernstein stopping rule",
arXiv:2502.01730v1.
"""

from math import log, sqrt, ceil


def eb_radius(n: int, R: float, var_biased: float, delta: float) -> float:
    """Compute empirical Bernstein radius per Eq. (4).

    The empirical Bernstein inequality states:
        |Ē_N - E| ≤ σ̄_N * sqrt(2*ln(3/δ)/N) + R * 3*ln(3/δ)/N =: ε_N

    where:
        σ̄_N² is the biased empirical variance (variance_biased)
        R = b - a is the range of the bounded random variable
        δ is the failure probability
        N is the number of samples

    Args:
        n: Number of samples (must be >= 1)
        R: Range of the bounded random variable (b - a)
        var_biased: Biased empirical variance (σ̄_N² = m2/N)
        delta: Failure probability (must be in (0, 1))

    Returns:
        The empirical Bernstein radius ε_n

    Raises:
        ValueError: If n < 1, R <= 0, delta not in (0, 1), or var_biased < 0
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if R <= 0:
        raise ValueError("R must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if var_biased < 0:
        raise ValueError("var_biased must be >= 0")

    # Pre-compute log term
    log_term = log(3.0 / delta)

    # First term: σ̄_N * sqrt(2*ln(3/δ)/N)
    sqrt_term = sqrt(2 * log_term / n)
    first_term = sqrt(var_biased) * sqrt_term

    # Second term: R * 3*ln(3/δ)/N
    second_term = R * 3 * log_term / n

    return first_term + second_term


def eb_radius_modified(
    n: int,
    R: float,
    var_biased: float,
    delta_k: float,
    alpha: float = 1.0,
) -> float:
    """Compute modified EB radius with mid-interval stopping.

    Uses the mid-interval modification from Algorithm 2, where:
        x = -α * ln(δ_k/3)

    The modified radius is:
        ε_n = σ̄_n * sqrt(2*x/n) + R * x/n

    When α=1, this tightens the bound at checkpoints by using δ_k instead
    of δ, where δ_k is the per-check confidence allocation.

    Args:
        n: Number of samples (must be >= 1)
        R: Range of the bounded random variable (b - a)
        var_biased: Biased empirical variance
        delta_k: Per-check failure probability (must be in (0, 3))
        alpha: Mid-interval tightness parameter (default 1.0)

    Returns:
        The modified empirical Bernstein radius ε_n

    Raises:
        ValueError: If parameters are invalid
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if R <= 0:
        raise ValueError("R must be > 0")
    if delta_k <= 0 or delta_k >= 3:
        raise ValueError("delta_k must be in (0, 3)")
    if var_biased < 0:
        raise ValueError("var_biased must be >= 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    # Compute x = -α * ln(δ_k/3)
    x = -alpha * log(delta_k / 3.0)

    # First term: σ̄_n * sqrt(2*x/n)
    first_term = sqrt(var_biased) * sqrt(2 * x / n)

    # Second term: R * x/n
    second_term = R * x / n

    return first_term + second_term


def geom_checkpoints(beta: float, n_min: int, n_max: int) -> list[int]:
    """Generate geometric checkpoints for EBS evaluation.

    Returns checkpoints at n_k = ⌈β^k⌉ for k = k0, k0+1, ... while ≤ n_max,
    where k0 = ⌈log_β(n_min)⌉ is the smallest integer such that β^k0 ≥ n_min.

    The last checkpoint is always n_max (if not already included).

    Args:
        beta: Geometric growth factor (> 1). Controls how frequently we check.
        n_min: Minimum number of samples before first check (typically 10)
        n_max: Maximum number of samples (Hoeffding cap)

    Returns:
        Sorted list of checkpoint shot counts

    Raises:
        ValueError: If beta <= 1, n_min < 1, or n_max < n_min
    """
    if beta <= 1:
        raise ValueError("beta must be > 1")
    if n_min < 1:
        raise ValueError("n_min must be >= 1")
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min")

    # Find k0: smallest k such that beta^k >= n_min
    if n_min == 1:
        k0 = 0
    else:
        k0 = int(ceil(log(n_min) / log(beta)))

    checkpoints: list[int] = []
    k = k0

    while True:
        n_k = int(ceil(beta**k))
        if n_k > n_max:
            break
        checkpoints.append(n_k)
        k += 1

    # Always include n_max as the final checkpoint if not already present
    if not checkpoints or checkpoints[-1] != n_max:
        checkpoints.append(n_max)

    return checkpoints


def ebs_delta_schedule(delta: float, K: int) -> list[float]:
    """Allocate failure probability uniformly over K checks.

    Per the paper's recommendation for finite K: distribute δ equally
    across all checks. Each check gets d_k = δ / K.

    Args:
        delta: Total failure probability (must be in (0, 1))
        K: Number of checks (must be >= 1)

    Returns:
        List of K per-check failure probabilities, all equal to delta/K

    Raises:
        ValueError: If delta not in (0, 1) or K < 1
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if K < 1:
        raise ValueError("K must be >= 1")

    d_k = delta / K
    return [d_k] * K
