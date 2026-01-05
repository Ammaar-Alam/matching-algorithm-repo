from __future__ import annotations

from math import sqrt, log
from typing import Tuple

EPS = 1e-9


def empirical_bernstein_radius(
    var_hat: float,
    n_eff: int,
    delta: float,
) -> float:
    """
    Empirical-Bernstein radius for a bounded mean in [0,1].
    """
    n = max(int(n_eff), 1)
    v = max(float(var_hat), 0.0)
    if delta <= 0.0 or delta >= 1.0:
        delta = 0.10
    t = log(2.0 / delta)
    return sqrt(2.0 * v * t / float(n)) + 3.0 * t / float(n)


def lcb_bernstein(
    mean: float,
    var_hat: float,
    n_eff: int,
    delta: float,
) -> float:
    r = empirical_bernstein_radius(var_hat, n_eff, delta)
    lo = float(mean) - r
    if lo < 0.0:
        return 0.0
    if lo > 1.0:
        return 1.0
    return lo
