from __future__ import annotations

from math import log
from typing import Dict, Iterable, Tuple


def idf_with_shrink(
    sums: Dict[str, float],
    counts: Dict[str, int],
    alpha: float = 20.0,
    p0: float = 0.5,
    beta: float = 1e-2,
) -> Dict[str, float]:
    """
    Compute IDF-style rarity weights with a simple Beta prior.
    sums[id]   = sum of similarities m_i over observed pairs
    counts[id] = number of pairs contributing to that sum
    """
    weights: Dict[str, float] = {}
    for qid, c in counts.items():
        n = int(c)
        if n <= 0:
            p_hat = 0.0
        else:
            p_hat = float(sums.get(qid, 0.0)) / float(max(n, 1))
        num = alpha * p0 + float(n) * p_hat
        den = alpha + float(n)
        p = num / den if den > 0.0 else p0
        weights[qid] = float(log((1.0 + beta) / (p + beta)))
    return weights


def accumulate_pair_sims(
    item_ids: Iterable[str],
    pairs: Iterable[Tuple[str, str, float]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Utility to accumulate per-item similarity sums and counts.
    pairs yields (item_id, key, m) where key is ignored here.
    """
    sums: Dict[str, float] = {qid: 0.0 for qid in item_ids}
    counts: Dict[str, int] = {qid: 0 for qid in item_ids}
    for qid, _, m in pairs:
        if qid not in sums:
            continue
        sums[qid] += float(m)
        counts[qid] += 1
    return sums, counts

