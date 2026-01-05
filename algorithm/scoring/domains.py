from __future__ import annotations

from typing import Dict, Tuple

EPS = 1e-9


def soft_cap_domain_contrib(
    contrib: Dict[str, float],
    rho: float,
) -> Tuple[Dict[str, float], float]:
    """
    Apply soft-cap to domain shares and renormalize.
    contrib[d] are non-negative domain contributions.
    """
    total = sum(contrib.values()) + EPS
    if total <= EPS:
        return {}, 0.0
    share = {d: c / total for d, c in contrib.items()}
    share_cap = {d: min(v, rho) for d, v in share.items()}
    excess = sum(share[d] - share_cap[d] for d in share if share[d] > share_cap[d])
    norm = sum(share_cap.values()) or 1.0
    share_final = {
        d: share_cap[d] + (share_cap[d] / norm) * excess for d in share_cap
    }
    scaled: Dict[str, float] = {}
    capped_total = 0.0
    for d, c in contrib.items():
        qd = share[d]
        qf = share_final[d]
        scale = (qf / qd) if qd > 0.0 else 0.0
        val = c * scale
        scaled[d] = val
        capped_total += val
    return scaled, capped_total

