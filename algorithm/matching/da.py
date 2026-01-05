from __future__ import annotations

from typing import Dict, List


def gale_shapley_da(
    proposers: Dict[str, List[str]],
    receivers: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Standard student-proposing DA for strict lists.
    proposers/receivers map ids to ranked lists (best first).
    Returns mapping proposer -> matched receiver.
    """
    pref_p = {p: list(lst) for p, lst in proposers.items()}
    rank_r: Dict[str, Dict[str, int]] = {
        r: {p: i for i, p in enumerate(lst)} for r, lst in receivers.items()
    }
    free = list(pref_p.keys())
    next_idx = {p: 0 for p in pref_p}
    match_r: Dict[str, str] = {}
    match_p: Dict[str, str] = {}
    while free:
        p = free.pop(0)
        lst = pref_p.get(p, [])
        i = next_idx.get(p, 0)
        if i >= len(lst):
            continue
        r = lst[i]
        next_idx[p] = i + 1
        if r not in rank_r:
            continue
        if r not in match_r:
            match_r[r] = p
            match_p[p] = r
        else:
            cur = match_r[r]
            if rank_r[r].get(p, 10**9) < rank_r[r].get(cur, 10**9):
                match_r[r] = p
                match_p[p] = r
                if cur in match_p:
                    del match_p[cur]
                free.append(cur)
            else:
                free.append(p)
    return match_p


def da_with_ties(
    proposers_tied: Dict[str, List[List[str]]],
    receivers_tied: Dict[str, List[List[str]]],
) -> Dict[str, str]:
    """
    Weakly stable DA variant for tied lists.
    Ties are represented as list of tie-classes in order.
    We expand ties to strict lists but treat all members of a tie
    at the same rank when receivers compare proposers.
    """
    proposers: Dict[str, List[str]] = {}
    for p, tiers in proposers_tied.items():
        linear: List[str] = []
        for tier in tiers:
            linear.extend(tier)
        proposers[p] = linear

    receivers: Dict[str, List[str]] = {}
    for r, tiers in receivers_tied.items():
        linear: List[str] = []
        for tier in tiers:
            linear.extend(tier)
        receivers[r] = linear

    return gale_shapley_da(proposers, receivers)

