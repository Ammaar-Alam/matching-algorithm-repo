from __future__ import annotations

import argparse, json, itertools, os, sys
from typing import Dict, Any

# ensure tiger-alg root on path for sibling package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


def lcb_from(shat: float, var: float, delta: float) -> float:
    if var <= 1e-12:
        return max(0.0, min(1.0, float(shat)))
    z = _z_for_delta(float(delta))
    from math import sqrt
    return max(0.0, min(1.0, float(shat) - z * sqrt(float(var))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--questions_json", required=True)
    ap.add_argument("--model", default=None, help="optional acceptability params json; defaults used if omitted")
    ap.add_argument("--weights_json", default=None, help="optional JSON domain weights to override defaults")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out", default="scores.csv")
    args = ap.parse_args()

    export_df = pd.read_csv(args.export_csv)
    meta = load_meta(args.meta_json)
    q47_opts = make_q47_options(args.questions_json)

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

    rows = []
    by_id = {str(r["participantId"]): r.to_dict() for _, r in export_df.iterrows()}
    ids = list(by_id.keys())
    for A, B in itertools.permutations(ids, 2):
        rowA = by_id[A]
        rowB = by_id[B]
        prA = extract_priorities_from_q47(rowA, q47_opts)
        prB = extract_priorities_from_q47(rowB, q47_opts)

        sA, vA, nA, _ = directional_stats(rowA, rowB, meta, kernels, imp_map, dom_w, prA, sims)
        sB, vB, nB, _ = directional_stats(rowB, rowA, meta, kernels, imp_map, dom_w, prB, sims)
        lA = lcb_from(sA, vA, args.delta)
        lB = lcb_from(sB, vB, args.delta)
        score = combine_scores(lA, lB)
        pct_geo = baseline_like_percent_from_lcbs(lA, lB)
        pct_both = both_pass_percent_from_lcbs(lA, lB)
        pct_sym = symmetric_percent_from_score(score)
        rows.append({
            "A": A, "B": B,
            "LCB_A": lA, "LCB_B": lB, "score": score,
            "pct_baseline_like": pct_geo,
            "pct_both_pass": pct_both,
            "pct_from_score_symmetric": pct_sym,
            "overlap_A": nA, "overlap_B": nB
        })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
