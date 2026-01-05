from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ensure algorithm/ root on path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.core import DEFAULT_DOMAIN_MULTIPLIER  # noqa: E402
from algorithms.best_stable_lp import best_stable_lp  # noqa: E402
from matching.blossom import max_weight_matching_exact  # noqa: E402
from matching.da import gale_shapley_da  # noqa: E402
from scoring.domains import soft_cap_domain_contrib  # noqa: E402
from scoring.lcb import lcb_bernstein  # noqa: E402
from scoring.rarity import idf_with_shrink  # noqa: E402


EPS = 1e-9
DEFAULT_RHO = 0.35
DEFAULT_TAU = 0.05
DEFAULT_MU = 0.03
DEFAULT_TOPK = 2
DEFAULT_DELTA = 0.10


@dataclass
class ItemMeta:
    id: str
    domain: str
    kind: str  # 'likert' | 'multiselect' | 'binary'
    rng: float


def _to_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        s = str(val).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


def _to_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val).strip()
    if s == "":
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        return [str(obj)]
    except Exception:
        return [s]


def build_items(meta: Dict[str, Dict[str, Any]]) -> List[ItemMeta]:
    items: List[ItemMeta] = []
    for qid, m in meta.items():
        rt = str(m.get("ResponseType", "Likert5"))
        dom = str(m.get("Domain", ""))
        kind = "likert"
        rng = 1.0
        if rt.startswith("Likert"):
            if "7" in rt:
                rng = 6.0
            elif "6" in rt:
                rng = 5.0
            else:
                rng = 4.0
            kind = "likert"
        elif rt.startswith("MC"):
            kind = "multiselect"
            rng = 1.0
        elif rt.lower().startswith("binary"):
            kind = "binary"
            rng = 1.0
        else:
            kind = "likert"
            rng = 4.0
        items.append(ItemMeta(id=qid, domain=dom, kind=kind, rng=rng))
    return items


def item_similarity(item: ItemMeta, xa: Any, xb: Any) -> float:
    if xa is None or xb is None:
        return math.nan
    if item.kind == "likert":
        ia = _to_int(xa)
        ib = _to_int(xb)
        if ia is None or ib is None:
            return math.nan
        return max(0.0, 1.0 - abs(ia - ib) / float(item.rng or 1.0))
    if item.kind == "multiselect":
        sa = set(_to_list(xa))
        sb = set(_to_list(xb))
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        if uni == 0:
            return 0.0
        return inter / float(uni)
    # binary
    ia = _to_int(xa)
    ib = _to_int(xb)
    if ia is None or ib is None:
        return math.nan
    return 1.0 if ia == ib else 0.0


def build_people(export_df: pd.DataFrame, items: List[ItemMeta]) -> Dict[str, Dict[str, Any]]:
    people: Dict[str, Dict[str, Any]] = {}
    for _, row in export_df.iterrows():
        pid = str(row.get("participantId"))
        if not pid or pid == "nan":
            continue
        resp: Dict[str, Any] = {}
        for it in items:
            col = f"SELF_{it.id}"
            resp[it.id] = row.get(col, None)
        people[pid] = resp
    return people


def compute_idf_weights(items: List[ItemMeta], people: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    ids = list(people.keys())
    n_ids = len(ids)
    sums: Dict[str, float] = {it.id: 0.0 for it in items}
    cnt: Dict[str, int] = {it.id: 0 for it in items}
    for i in range(n_ids):
        pa = people[ids[i]]
        for j in range(i + 1, n_ids):
            pb = people[ids[j]]
            for it in items:
                m = item_similarity(it, pa.get(it.id), pb.get(it.id))
                if not math.isnan(m):
                    sums[it.id] += m
                    cnt[it.id] += 1
    return idf_with_shrink({it.id: sums[it.id] for it in items}, {it.id: cnt[it.id] for it in items})


def directed_score(
    items: List[ItemMeta],
    theta: Dict[str, float],
    w_idf: Dict[str, float],
    a_resp: Dict[str, Any],
    b_resp: Dict[str, Any],
    rho: float,
    delta: float,
) -> Tuple[Dict[str, float], int, float, float]:
    num: Dict[str, float] = {}
    den: Dict[str, float] = {}
    # accumulators for weighted EB statistics over item-level similarities m in [0,1]
    sum_w = 0.0
    sum_w2 = 0.0
    sum_wm = 0.0
    sum_w2m = 0.0
    sum_w2m2 = 0.0
    overlap = 0
    for it in items:
        m = item_similarity(it, a_resp.get(it.id), b_resp.get(it.id))
        if math.isnan(m):
            continue
        overlap += 1
        w = w_idf.get(it.id, 1.0)
        d = it.domain
        num[d] = num.get(d, 0.0) + w * m
        den[d] = den.get(d, 0.0) + w
        sum_w += w
        w2 = w * w
        sum_w2 += w2
        sum_wm += w * m
        sum_w2m += w2 * m
        sum_w2m2 += w2 * m * m
    contrib: Dict[str, float] = {}
    for d, v in num.items():
        s_d = v / float(den.get(d, 0.0) + EPS)
        c = (theta.get(d, 0.0) or 0.0) * s_d
        if c <= 0.0:
            continue
        contrib[d] = c
    if not contrib:
        return {}, overlap, 0.0, 0.0
    scaled, capped_total = soft_cap_domain_contrib(contrib, rho)
    if capped_total <= 0.0:
        return scaled, overlap, 0.0, 0.0
    # effective sample size for weighted items
    if sum_w2 <= 0.0:
        n_eff = max(overlap, 1)
    else:
        n_eff = int((sum_w * sum_w) / max(sum_w2, EPS))
        if n_eff <= 0:
            n_eff = max(overlap, 1)
    # weighted empirical variance of item-level similarities around their weighted mean
    if sum_w2 > 0.0 and sum_w > 0.0 and overlap > 0:
        mu_items = sum_wm / sum_w
        # Σ w^2 (m - μ)^2 = Σ w^2 m^2 - 2 μ Σ w^2 m + μ^2 Σ w^2
        var_num = sum_w2m2 - 2.0 * mu_items * sum_w2m + (mu_items * mu_items) * sum_w2
        var_hat = max(0.0, var_num) / max(sum_w2, EPS)
    else:
        var_hat = 0.0
    # normalize capped_total into [0,1] for the bound
    mu = max(0.0, min(1.0, capped_total))
    lcb = lcb_bernstein(mu, var_hat, n_eff, delta)
    return scaled, overlap, lcb, mu


def harmonic_mean(a: float, b: float) -> float:
    denom = a + b + EPS
    if denom <= 0.0:
        return 0.0
    return (2.0 * a * b) / denom


def build_preference_lists(
    ids: List[str],
    items: List[ItemMeta],
    theta: Dict[str, float],
    w_idf: Dict[str, float],
    people: Dict[str, Dict[str, Any]],
    rho: float,
    delta: float,
    tau: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[List[str]]], Dict[str, Dict[str, float]]]:
    lcb: Dict[str, Dict[str, float]] = {pid: {} for pid in ids}
    rad: Dict[str, Dict[str, float]] = {pid: {} for pid in ids}
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                continue
            per_dom, overlap, l, mu = directed_score(
                items,
                theta,
                w_idf,
                people[a],
                people[b],
                rho,
                delta,
            )
            if overlap <= 0:
                continue
            lcb[a][b] = l
            rad[a][b] = max(0.0, mu - l)
    lists: Dict[str, List[List[str]]] = {}
    for a in ids:
        rows = [
            (b, l)
            for b, l in lcb[a].items()
            if b != a and math.isfinite(l) and l > 0.0
        ]
        rows.sort(key=lambda t: t[1], reverse=True)
        ties: List[List[str]] = []
        bucket: List[str] = []
        last_l = None
        last_r = None
        for b, l in rows:
            r = rad.get(a, {}).get(b, 0.0)
            if last_l is None:
                bucket = [b]
                last_l = l
                last_r = r
                continue
            gap = last_l - l
            thr = tau * (last_r + r if (last_r + r) > 0.0 else 1.0)
            if gap < thr:
                bucket.append(b)
            else:
                if bucket:
                    ties.append(bucket)
                bucket = [b]
                last_l = l
                last_r = r
        if bucket:
            ties.append(bucket)
        lists[a] = ties
    return lcb, lists, rad


def pair_scores(
    ids: List[str],
    lcb: Dict[str, Dict[str, float]],
    lists: Dict[str, List[List[str]]],
    mu: float,
    topk: int,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    top_sets: Dict[str, set[str]] = {}
    for pid, tie_list in lists.items():
        flat: List[str] = []
        for grp in tie_list[: max(0, topk)]:
            flat.extend(grp)
        top_sets[pid] = set(flat)
    scores: Dict[Tuple[str, str], float] = {}
    base_hm: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            la = lcb.get(a, {}).get(b, 0.0)
            lb = lcb.get(b, {}).get(a, 0.0)
            hm = harmonic_mean(la, lb)
            base_hm[(a, b)] = hm
            base_hm[(b, a)] = hm
            if a in top_sets.get(b, set()) and b in top_sets.get(a, set()):
                val = hm * (1.0 + mu)
            else:
                val = hm
            scores[(a, b)] = val
            scores[(b, a)] = val
    return scores, base_hm


def build_da_sides(
    ids: List[str],
    lists_linear: Dict[str, List[str]],
    lcb: Dict[str, Dict[str, float]],
    sides: Dict[str, List[str]] | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Bipartite split for DA:
    - if sides is provided, use its 'left' and 'right' lists (filtered to ids);
    - otherwise, use first half as proposers, second half as receivers.
    Receivers rank proposers by their own directional LCB.
    """
    if sides and isinstance(sides.get("left"), list) and isinstance(sides.get("right"), list):
        left_ids = [i for i in sides["left"] if i in ids]
        right_ids = [i for i in sides["right"] if i in ids]
    else:
        n = len(ids)
        mid = n // 2
        left_ids = ids[:mid]
        right_ids = ids[mid:]

    proposers: Dict[str, List[str]] = {
        p: [r for r in lists_linear.get(p, []) if r in right_ids] for p in left_ids
    }

    receivers: Dict[str, List[str]] = {}
    for r in right_ids:
        cand = [p for p in left_ids if r in lists_linear.get(p, [])]
        receivers[r] = sorted(
            cand, key=lambda p: lcb.get(r, {}).get(p, 0.0), reverse=True
        )
    return proposers, receivers


def summarize_true_partner_ranks(
    couples_df: pd.DataFrame,
    ids: List[str],
    lcb: Dict[str, Dict[str, float]],
    scores: Dict[Tuple[str, str], float],
) -> Dict[str, Any]:
    per_person: List[Dict[str, Any]] = []
    hit1 = hit2 = hit3 = 0
    ranks: List[int] = []
    id_set = set(ids)
    for _, row in couples_df.iterrows():
        a = str(row.get("partner_a_id"))
        b = str(row.get("partner_b_id"))
        if a not in id_set or b not in id_set:
            continue
        for anchor, target in ((a, b), (b, a)):
            others = [x for x in ids if x != anchor]
            # rank candidates for this anchor by directional LCB (not symmetric pair score)
            vals = [(o, lcb.get(anchor, {}).get(o, 0.0)) for o in others]
            vals.sort(key=lambda t: t[1], reverse=True)
            rank = next((idx + 1 for idx, (o, _) in enumerate(vals) if o == target), None)
            if rank is None:
                continue
            gap = None
            if len(vals) >= 2:
                best = vals[0][1]
                second = vals[1][1]
                gap = float(best - second)
            tag = "fragile"
            if gap is not None and gap >= 0.20:
                tag = "robust"
            elif gap is not None and gap >= 0.10:
                tag = "moderate"
            per_person.append(
                {
                    "anchor": anchor,
                    "partner": target,
                    "rank": int(rank),
                    "top_gap": float(gap) if gap is not None else math.nan,
                    "robustness": tag,
                }
            )
            ranks.append(int(rank))
            if rank == 1:
                hit1 += 1
            if rank <= 2:
                hit2 += 1
            if rank <= 3:
                hit3 += 1
    summary: Dict[str, Any] = {}
    n = len(ranks)
    if n > 0:
        arr = np.array(ranks, dtype=float)
        summary["n_people"] = int(n)
        summary["hit_at_1"] = float(hit1) / float(n)
        summary["hit_at_2"] = float(hit2) / float(n)
        summary["hit_at_3"] = float(hit3) / float(n)
        summary["median_rank"] = float(np.median(arr))
        summary["mean_rank"] = float(np.mean(arr))
    else:
        summary["n_people"] = 0
    return {"per_person": per_person, "summary": summary}


def count_blocking_pairs(
    matching: Dict[str, str],
    ids: List[str],
    lists_linear: Dict[str, List[str]],
    scores: Dict[Tuple[str, str], float],
    left_ids: List[str] | None = None,
    right_ids: List[str] | None = None,
    tie_tiers: Dict[str, Dict[str, int]] | None = None,
) -> int:
    id_set = set(ids)
    block = 0
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            if a not in id_set or b not in id_set:
                continue
            # if a bipartition is provided, only consider cross-partition pairs
            if left_ids is not None and right_ids is not None:
                in_left_a = a in left_ids
                in_left_b = b in left_ids
                in_right_a = a in right_ids
                in_right_b = b in right_ids
                if not (
                    (in_left_a and in_right_b) or (in_left_b and in_right_a)
                ):
                    continue
            # only consider pairs both sides list as acceptable
            if b not in lists_linear.get(a, []) or a not in lists_linear.get(b, []):
                continue
            ma = matching.get(a)
            mb = matching.get(b)
            if ma == b or mb == a:
                continue
            prefs_a = lists_linear.get(a, [])
            prefs_b = lists_linear.get(b, [])
            # strictly better under tiers: earlier tier index; same tier is not better
            def better(anchor: str, target: str, current: str | None) -> bool:
                if current is None:
                    return True
                if tie_tiers and anchor in tie_tiers:
                    tiers = tie_tiers[anchor]
                    t_t = tiers.get(target)
                    t_c = tiers.get(current)
                    if t_t is None:
                        return False
                    if t_c is None:
                        return True
                    return t_t < t_c
                # fallback: list index comparison
                if target not in prefs_a and current not in prefs_a:
                    return False
                if target in prefs_a and current not in prefs_a:
                    return True
                if target not in prefs_a:
                    return False
                return prefs_a.index(target) < prefs_a.index(current)

            better_for_a = better(a, b, ma)
            # reuse helper for b's perspective
            prefs_a = prefs_b  # for fallback index case
            better_for_b = better(b, a, mb)
            if better_for_a and better_for_b:
                block += 1
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_csv", required=True)
    ap.add_argument("--couples_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--outdir", default="out_merged")
    ap.add_argument("--rho", type=float, default=DEFAULT_RHO)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--mu", type=float, default=DEFAULT_MU)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--sides_json", default=None, help="Optional JSON with explicit left/right partition")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    export_df = pd.read_csv(args.export_csv)
    couples_df = pd.read_csv(args.couples_csv)
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    items = build_items(meta)
    people = build_people(export_df, items)
    ids = sorted(people.keys())
    if len(ids) == 0:
        print("no participants found in export.csv")
        return

    # normalize domain weights so they sum to 1
    raw_theta = dict(DEFAULT_DOMAIN_MULTIPLIER)
    theta_sum = sum(max(0.0, v) for v in raw_theta.values()) or 1.0
    theta = {d: max(0.0, v) / theta_sum for d, v in raw_theta.items()}
    w_idf = compute_idf_weights(items, people)

    lcb, tie_lists, rad = build_preference_lists(
        ids,
        items,
        theta,
        w_idf,
        people,
        rho=args.rho,
        delta=args.delta,
        tau=args.tau,
    )
    scores, base_hm = pair_scores(ids, lcb, tie_lists, mu=args.mu, topk=args.topk)

    linear_lists: Dict[str, List[str]] = {}
    for pid, groups in tie_lists.items():
        flat: List[str] = []
        for grp in groups:
            flat.extend(sorted(grp))
        linear_lists[pid] = flat

    # optional explicit bipartition
    sides_data: Dict[str, List[str]] | None = None
    if args.sides_json:
        try:
            with open(args.sides_json, "r", encoding="utf-8") as sf:
                raw = json.load(sf)
            if isinstance(raw, dict):
                left = raw.get("left") or []
                right = raw.get("right") or []
                if isinstance(left, list) and isinstance(right, list):
                    sides_data = {"left": [str(x) for x in left], "right": [str(x) for x in right]}
        except Exception:
            sides_data = None

    # bipartite DA and best-stable LP
    proposers, receivers = build_da_sides(ids, linear_lists, lcb, sides_data)
    da_match = gale_shapley_da(proposers, receivers)
    lp_match = best_stable_lp(
        proposers,
        receivers,
        scores,
        status_path=os.path.join(args.outdir, "solver.txt"),
    )
    blossom_match = max_weight_matching_exact(ids, scores)

    pair_rows: List[Dict[str, Any]] = []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            l_ab = float(lcb.get(a, {}).get(b, 0.0))
            l_ba = float(lcb.get(b, {}).get(a, 0.0))
            r_ab = float(rad.get(a, {}).get(b, 0.0))
            r_ba = float(rad.get(b, {}).get(a, 0.0))
            hm = float(base_hm.get((a, b), 0.0))
            hm_mut = float(scores.get((a, b), 0.0))
            pair_rows.append(
                {
                    "A": a,
                    "B": b,
                    "LCB_A": l_ab,
                    "RAD_A": r_ab,
                    "LCB_B": l_ba,
                    "RAD_B": r_ba,
                    "HM": hm,
                    "HM_mutual": hm_mut,
                }
            )
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(args.outdir, "pair_scores_merged.csv"),
        index=False,
    )

    with open(os.path.join(args.outdir, "lists_tied.json"), "w", encoding="utf-8") as f:
        json.dump(tie_lists, f, indent=2)

    with open(os.path.join(args.outdir, "matching_da.json"), "w", encoding="utf-8") as f:
        json.dump(da_match, f, indent=2)

    with open(os.path.join(args.outdir, "matching_best_stable_lp.json"), "w", encoding="utf-8") as f:
        json.dump(lp_match, f, indent=2)

    with open(os.path.join(args.outdir, "matching_max_weight.json"), "w", encoding="utf-8") as f:
        json.dump(blossom_match, f, indent=2)

    eval_obj = summarize_true_partner_ranks(couples_df, ids, lcb, scores)
    with open(os.path.join(args.outdir, "summary_ranks.json"), "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=2)

    summ = eval_obj.get("summary", {})
    n_people = int(summ.get("n_people", 0) or 0)
    print(f"merged pipeline over {len(ids)} participants, {n_people} with known partners")
    if n_people > 0:
        print(
            f"Hit@1={summ.get('hit_at_1', 0.0):.3f}, "
            f"Hit@2={summ.get('hit_at_2', 0.0):.3f}, "
            f"Hit@3={summ.get('hit_at_3', 0.0):.3f}, "
            f"median rank={summ.get('median_rank', float('nan')):.2f}"
        )
    # basic stability diagnostics
    bp_da = count_blocking_pairs(da_match, ids, linear_lists, scores)
    bp_lp = count_blocking_pairs(lp_match, ids, linear_lists, scores)
    bp_mw = count_blocking_pairs(blossom_match, ids, linear_lists, scores)

    print(f"DA matching → {os.path.join(args.outdir, 'matching_da.json')}  (blocking pairs={bp_da})")
    print(f"best-stable LP matching → {os.path.join(args.outdir, 'matching_best_stable_lp.json')}  (blocking pairs={bp_lp})")
    print(f"max-weight matching → {os.path.join(args.outdir, 'matching_max_weight.json')}  (blocking pairs={bp_mw})")


if __name__ == "__main__":
    main()
