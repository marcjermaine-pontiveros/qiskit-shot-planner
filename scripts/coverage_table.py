"""Empirical coverage of the geometric-checkpoint rule, with Clopper-Pearson bounds.

For each variance level, runs the EBS-geom stopper over all seeds x trials (the
SAME pipeline as the sample-complexity table, imported from regen_sim_tables),
counts how often the (eps, delta) guarantee held (|estimate - mu| <= eps),
pools the trials, and reports the exact one-sided 95% Clopper-Pearson lower
bound on the true coverage. Near p_hat = 1 this is the correct interval; the
Wald normal approximation would give a degenerate zero-width interval there.

    python3 scripts/coverage_table.py            # 5 seeds x 500 = 2500 trials/level
"""

from __future__ import annotations

import argparse
import os
import sys
from math import sqrt

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_sim_tables import EPS, DELTA, SIGMA2_GRID, THESIS_SEEDS, _bernoulli_device, _ebs

CONF = 0.95  # one-sided confidence for the Clopper-Pearson lower bound


def cp_lower(k: int, n: int, conf: float = CONF) -> float:
    """Exact one-sided Clopper-Pearson lower bound on p given k/n successes."""
    if k == 0:
        return 0.0
    return float(beta.ppf(1.0 - conf, k, n - k + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=500, help="trials per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=THESIS_SEEDS)
    args = ap.parse_args()

    n_total = args.trials * len(args.seeds)
    print(f"Coverage validation: eps={EPS}, delta={DELTA}, target coverage >= {100*(1-DELTA):.0f}%")
    print(f"{n_total} trials/level ({len(args.seeds)} seeds x {args.trials}); "
          f"one-sided {int(CONF*100)}% Clopper-Pearson lower bound\n")
    print(f"{'sigma^2':>8} {'|mu|':>6} {'trials':>7} {'fails':>6} {'coverage%':>10} {'CP lower%':>10}")
    rows = []
    for s2 in SIGMA2_GRID:
        mu = sqrt(max(0.0, 1.0 - s2))
        k = 0
        for master in args.seeds:
            mrng = np.random.default_rng(master)
            for _ in range(args.trials):
                sd = int(mrng.integers(0, 2**31))
                r = _ebs(_bernoulli_device(mu, sd))
                if abs(r.estimate - mu) <= EPS:
                    k += 1
        phat = 100.0 * k / n_total
        lo = 100.0 * cp_lower(k, n_total)
        print(f"{s2:8.2f} {mu:6.3f} {n_total:7d} {n_total-k:6d} {phat:9.2f}% {lo:9.2f}%")
        rows.append((s2, mu, n_total, n_total - k, phat, lo))
    return rows


if __name__ == "__main__":
    main()
