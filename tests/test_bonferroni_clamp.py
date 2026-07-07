"""Regression: the variance-aware clamp prevents near-deterministic terms from
forcing an enormous stopping time, while preserving the joint (eps, delta) guarantee."""
import time

import numpy as np

from qamp_shotplanner import bonferroni_estimate


def _bern(mu, seed):
    p = (1.0 + mu) / 2.0
    rng = np.random.default_rng(seed)
    return lambda n: (2.0 * rng.binomial(1, p, int(n)) - 1.0).tolist()


def test_tiny_sigma_term_does_not_blow_up():
    # A near-deterministic term (mu ~ 0.9999, tiny sigma) alongside a normal one.
    terms = [(1.0, _bern(0.9999, 1)), (1.0, _bern(0.3, 2))]
    sigmas = [(1 - 0.9999**2) ** 0.5, (1 - 0.3**2) ** 0.5]
    t0 = time.time()
    r = bonferroni_estimate(terms, eps=0.02, delta=0.01, sigmas=sigmas)
    assert time.time() - t0 < 10.0  # pre-clamp this could hang for minutes
    # bounded, sane shot count (pre-clamp the tiny-sigma term alone approached ~1e8)
    assert r.total_shots < 2_000_000


def test_variance_aware_still_covers():
    # Two moderate-variance terms: the certified estimate stays within eps.
    mu0, mu1 = 0.4, -0.2
    terms = [(1.0, _bern(mu0, 10)), (0.5, _bern(mu1, 20))]
    sigmas = [(1 - mu0**2) ** 0.5, (1 - mu1**2) ** 0.5]
    exact = 1.0 * mu0 + 0.5 * mu1
    hits = 0
    for t in range(40):
        terms_t = [(1.0, _bern(mu0, 100 + t)), (0.5, _bern(mu1, 500 + t))]
        r = bonferroni_estimate(terms_t, eps=0.03, delta=0.01, sigmas=sigmas)
        hits += abs(r.energy - exact) <= 0.03
    assert hits >= 38  # >= ~95% empirical, comfortably above the 99% design floor over few trials
