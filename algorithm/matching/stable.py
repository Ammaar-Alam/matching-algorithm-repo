from __future__ import annotations

import argparse, json
from typing import Dict, List


def gale_shapley(proposers: Dict[str, List[str]], receivers: Dict[str, List[str]]) -> Dict[str, str]:
    # simple gs with ties broken by list order
    pref_p = {p: list(lst) for p, lst in proposers.items()}
    rank_r: Dict[str, Dict[str, int]] = {r: {p: i for i, p in enumerate(lst)} for r, lst in receivers.items()}
    free = list(pref_p.keys())
    next_idx = {p: 0 for p in pref_p}
    match_r: Dict[str, str] = {}
    match_p: Dict[str, str] = {}

    while free:
        p = free.pop(0)
        lst = pref_p.get(p, [])
        if next_idx[p] >= len(lst):
            continue
        r = lst[next_idx[p]]
        next_idx[p] += 1
        if r not in rank_r:
            continue
        if r not in match_r:
            match_r[r] = p
            match_p[p] = r
        else:
            cur = match_r[r]
            # lower index = higher preference
            if rank_r[r].get(p, 10**9) < rank_r[r].get(cur, 10**9):
                match_r[r] = p
                match_p[p] = r
                if cur in match_p:
                    del match_p[cur]
                free.append(cur)
            else:
                free.append(p)
    return match_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True, help="json mapping id->ranked list")
    ap.add_argument("--mode", choices=["bipartite"], default="bipartite")
    ap.add_argument("--out", default="matching.json")
    args = ap.parse_args()

    with open(args.lists, "r") as f:
        lists = json.load(f)
    # single population bipartite split by even/odd index
    ids = sorted(lists.keys())
    left = {i: lists[i] for i in ids[::2]}
    right = {i: lists[i] for i in ids[1::2]}
    # build receiver rankings by reversing proposer lists as a crude proxy
    receivers: Dict[str, List[str]] = {r: [] for r in right.keys()}
    for p, lst in left.items():
        for r in lst:
            if r in receivers:
                receivers[r].append(p)
    m = gale_shapley(left, receivers)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

