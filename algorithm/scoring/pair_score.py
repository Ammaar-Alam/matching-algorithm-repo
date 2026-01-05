from __future__ import annotations

from math import log, exp, sqrt


def logit(p: float, eps: float = 1e-6) -> float:
    if p <= eps:
        return -20.0
    if p >= 1.0 - eps:
        return 20.0
    return float(log(p/(1.0-p)))


def combine_scores(lcbA: float, lcbB: float) -> float:
    return logit(lcbA) + logit(lcbB)


def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = exp(-x)
        return 1.0/(1.0+z)
    else:
        z = exp(x)
        return z/(1.0+z)


# ---------- percent-style helpers ----------
def baseline_like_percent_from_lcbs(lcbA: float, lcbB: float) -> float:
    """Baseline-style % using geometric mean of directional LCBs (pre-penalty)."""
    a = min(max(float(lcbA), 0.0), 1.0)
    b = min(max(float(lcbB), 0.0), 1.0)
    return 100.0 * sqrt(a * b)


def both_pass_percent_from_lcbs(lcbA: float, lcbB: float) -> float:
    """Independence-proxy joint probability that both directions pass."""
    a = min(max(float(lcbA), 0.0), 1.0)
    b = min(max(float(lcbB), 0.0), 1.0)
    return 100.0 * (a * b)


def symmetric_percent_from_score(score: float) -> float:
    """
    Quick approximation when only the summed log-odds score is available:
    assumes LCB_A ≈ LCB_B so that score ≈ 2*logit(p).
    """
    return 100.0 * _sigmoid(0.5 * float(score))
