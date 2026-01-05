from __future__ import annotations

from typing import Dict, Iterable, Tuple


def max_weight_matching_exact(
    ids: Iterable[str],
    weights: Dict[Tuple[str, str], float],
) -> Dict[str, str]:
    """
    Exact max-weight matching wrapper.
    Tries networkx; falls back to greedy if unavailable.
    """
    try:
        import networkx as nx  # type: ignore
    except Exception:
        return _greedy_fallback(ids, weights)

    g = nx.Graph()
    for a in ids:
        g.add_node(a)
    for (a, b), w in weights.items():
        if a >= b or w <= 0.0:
            continue
        g.add_edge(a, b, weight=float(w))
    m = nx.algorithms.matching.max_weight_matching(g, maxcardinality=False)
    out: Dict[str, str] = {}
    for a, b in m:
        out[str(a)] = str(b)
        out[str(b)] = str(a)
    return out


def _greedy_fallback(
    ids: Iterable[str],
    weights: Dict[Tuple[str, str], float],
) -> Dict[str, str]:
    used: set[str] = set()
    out: Dict[str, str] = {}
    pairs = [
        (a, b, w)
        for (a, b), w in weights.items()
        if a < b and w > 0.0
    ]
    pairs.sort(key=lambda t: t[2], reverse=True)
    for a, b, w in pairs:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out[a] = b
        out[b] = a
    return out

