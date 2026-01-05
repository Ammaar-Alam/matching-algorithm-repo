from __future__ import annotations

import argparse, json, random, os, sys
from typing import Dict, Any, List, Tuple

# ensure tiger-alg root on path for sibling package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from algorithms.core import extract_priorities_from_q47
from models.accept_kernels import AcceptabilityParams
from models.domain_weights import DomainWeights
from models.importance_map import ImportanceMap
from models.mc_similarity import build_mc_similarity
from scoring.directional import directional_stats, _z_for_delta
from scoring.pair_score import (
    combine_scores,
    baseline_like_percent_from_lcbs,
    both_pass_percent_from_lcbs,
    symmetric_percent_from_score,
)


def load_meta(p: str) -> Dict[str, Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def make_q47_options(questions_json_path: str) -> Dict[str, str]:
    with open(questions_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    d = {}
    for q in items:
        if q.get("id") == "Q47":
            for idx, label in enumerate(q.get("options", []), start=1):
                d[str(idx)] = label
    return d


def auc_true_vs_random(true_scores: List[float], rnd_scores: List[float]) -> float:
    x = np.array(true_scores + rnd_scores, dtype=float)
    y = np.array([1]*len(true_scores) + [0]*len(rnd_scores), dtype=int)
    if len(true_scores) == 0 or len(rnd_scores) == 0:
        return float("nan")
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=float)
    i = 0; r = 1.0
    while i < len(order):
        j = i
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (r + (r + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg
        r += (j - i); i = j
    rank_sum = ranks[y == 1].sum()
    pos = y.sum(); neg = len(y) - pos
    U = rank_sum - pos * (pos + 1) / 2.0
    return float(U / (pos * neg)) if pos > 0 and neg > 0 else float("nan")


def lcb_from(shat: float, var: float, delta: float) -> float:
    if var <= 1e-12:
        return max(0.0, min(1.0, float(shat)))
    z = _z_for_delta(float(delta))
    from math import sqrt
    return max(0.0, min(1.0, float(shat) - z * sqrt(float(var))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_csv", required=True)
    ap.add_argument("--couples_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--questions_json", required=True)
    ap.add_argument("--out", default="out/summary_pam.csv")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--nonpartner_samples", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--bootstrap", type=int, default=0, help="paired bootstrap reps for CIs (0=off)")
    ap.add_argument("--permute", type=int, default=0, help="permutation reps for null p-value (0=off)")
    ap.add_argument("--weights_json", default=None, help="Optional JSON with domain weights to override defaults (keys=Domains)")
    args = ap.parse_args()

    export_df = pd.read_csv(args.export_csv)
    couples_df = pd.read_csv(args.couples_csv)
    meta = load_meta(args.meta_json)
    q47 = make_q47_options(args.questions_json)

    kernels = AcceptabilityParams.load(args.model) if args.model else AcceptabilityParams.from_defaults(meta)
    imp_map = ImportanceMap.from_defaults()
    dom_w = DomainWeights.from_defaults()
    if args.weights_json:
        with open(args.weights_json,'r',encoding='utf-8') as f:
            w_raw=json.load(f)
        cand={k: float(v) for k,v in w_raw.items() if k in dom_w.m}
        if cand:
            dom_w = DomainWeights(m=cand, psi=dom_w.psi)
    sims = build_mc_similarity(args.export_csv, args.questions_json)

    by_id = {str(r["participantId"]): r.to_dict() for _, r in export_df.iterrows()}
    ids = list(by_id.keys())

    rng = random.Random(123)
    true_scores = []
    rnd_scores = []

    per_rows = []
    for _, c in couples_df.iterrows():
        A = str(c["partner_a_id"]) ; B = str(c["partner_b_id"])
        if A not in ids or B not in ids:
            continue
        rowA = by_id[A] ; rowB = by_id[B]
        prA = extract_priorities_from_q47(rowA, q47)
        prB = extract_priorities_from_q47(rowB, q47)

        sA, vA, nA, _ = directional_stats(rowA, rowB, meta, kernels, imp_map, dom_w, prA, sims)
        sB, vB, nB, _ = directional_stats(rowB, rowA, meta, kernels, imp_map, dom_w, prB, sims)
        lA = lcb_from(sA, vA, args.delta)
        lB = lcb_from(sB, vB, args.delta)
        score = combine_scores(lA, lB)
        true_scores.append(score)
        per_rows.append({
            "A": A, "B": B,
            "LCB_A": lA, "LCB_B": lB, "score": score,
            "pct_baseline_like": baseline_like_percent_from_lcbs(lA, lB),
            "pct_both_pass": both_pass_percent_from_lcbs(lA, lB),
            "pct_from_score_symmetric": symmetric_percent_from_score(score),
            "overlap_A": nA, "overlap_B": nB
        })

        # sample non‑partners for A
        cand = [x for x in ids if x not in (A, B)]
        k = min(len(cand), max(0, int(args.nonpartner_samples)))
        for D in rng.sample(cand, k):
            rowD = by_id[D]
            prD = extract_priorities_from_q47(rowD, q47)
            sAD, vAD, _, _ = directional_stats(rowA, rowD, meta, kernels, imp_map, dom_w, prA, sims)
            sDA, vDA, _, _ = directional_stats(rowD, rowA, meta, kernels, imp_map, dom_w, prD, sims)
            lAD = lcb_from(sAD, vAD, args.delta)
            lDA = lcb_from(sDA, vDA, args.delta)
            rnd_scores.append(combine_scores(lAD, lDA))

    auc = auc_true_vs_random(true_scores, rnd_scores)
    lift = (float(np.mean(true_scores)) - float(np.mean(rnd_scores))) if (true_scores and rnd_scores) else float("nan")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pairs_path = args.out.replace(".csv",".pairs.csv")
    pd.DataFrame(per_rows).to_csv(pairs_path, index=False)

    # Optional paired bootstrap for CIs
    ci_auc = (float('nan'), float('nan'))
    ci_lift = (float('nan'), float('nan'))
    if args.bootstrap and len(true_scores) > 0 and len(rnd_scores) > 0:
        rng = np.random.default_rng(123)
        aucs = [] ; lifts = []
        for _ in range(int(args.bootstrap)):
            idx = rng.integers(0, len(per_rows), size=len(per_rows))
            ts = [ per_rows[i]["score"] for i in idx ]
            # sample negatives with replacement, same size as original rnd_scores
            rs = [ rnd_scores[j] for j in rng.integers(0, len(rnd_scores), size=len(rnd_scores)) ]
            aucs.append(auc_true_vs_random(ts, rs))
            lifts.append(float(np.mean(ts)) - float(np.mean(rs)))
        def pct(a, p):
            x = np.array(a, dtype=float)
            return float(np.nanpercentile(x, p))
        ci_auc = (pct(aucs, 2.5), pct(aucs, 97.5))
        ci_lift = (pct(lifts, 2.5), pct(lifts, 97.5))

    # Optional permutation (shuffle partners) for null p-value on AUC
    p_perm = float('nan')
    if args.permute and len(true_scores) > 0 and len(rnd_scores) > 0:
        rng = np.random.default_rng(123)
        count = 0
        ids = list(by_id.keys())
        obs_auc = auc_true_vs_random(true_scores, rnd_scores)
        for _ in range(int(args.permute)):
            # random partner assignment: pair A with random B != A
            perm_scores = []
            for _, c in couples_df.iterrows():
                A = str(c["partner_a_id"])
                cand = [u for u in ids if u != A]
                if not cand: continue
                B = rng.choice(cand)
                rowA = by_id.get(A); rowB = by_id.get(B)
                if rowA is None or rowB is None: continue
                prA = extract_priorities_from_q47(rowA, q47)
                prB = extract_priorities_from_q47(rowB, q47)
                sA, vA, _, _ = directional_stats(rowA, rowB, meta, kernels, imp_map, dom_w, prA, sims)
                sB, vB, _, _ = directional_stats(rowB, rowA, meta, kernels, imp_map, dom_w, prB, sims)
                from math import sqrt
                lA = lcb_from(sA, vA, args.delta)
                lB = lcb_from(sB, vB, args.delta)
                perm_scores.append(combine_scores(lA, lB))
            # compare permuted AUC to observed using same rnd_scores
            a = auc_true_vs_random(perm_scores, rnd_scores)
            if a >= obs_auc:
                count += 1
        p_perm = (count + 1) / (args.permute + 1)

    with open(args.out, "w") as f:
        f.write("metric,value\n")
        f.write(f"auc,{auc}\n")
        f.write(f"lift,{lift}\n")
        f.write(f"auc_ci_low,{ci_auc[0]}\n")
        f.write(f"auc_ci_high,{ci_auc[1]}\n")
        f.write(f"lift_ci_low,{ci_lift[0]}\n")
        f.write(f"lift_ci_high,{ci_lift[1]}\n")
        f.write(f"perm_p_auc,{p_perm}\n")
    print({"auc": auc, "lift": lift, "pairs": len(per_rows)})


if __name__ == "__main__":
    main()
