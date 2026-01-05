import json, argparse, random
import os, sys
# Ensure the repo root for tiger-alg is on sys.path so we can import sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms.core import final_match, extract_priorities_from_q47, DEFAULT_DOMAIN_MULTIPLIER
from algorithms.variants import VARIANTS

def load_meta(p:str)->Dict[str,Dict[str,Any]]:
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def load_df(p:str)->pd.DataFrame:
    return pd.read_csv(p)

def make_q47_options(questions_json_path:str)->Dict[str,str]:
    # Map Q47 option value -> label (so we can map to domains)
    with open(questions_json_path,"r",encoding="utf-8") as f:
        items=json.load(f)
    d={}
    for q in items:
        if q.get("id")=="Q47":
            for idx,label in enumerate(q.get("options",[]), start=1):
                d[str(idx)] = label
    return d

def auc_true_vs_random(true_scores:List[float], rnd_scores:List[float])->float:
    # Mann–Whitney U <-> AUC
    x = np.array(true_scores + rnd_scores, dtype=float)
    y = np.array([1]*len(true_scores) + [0]*len(rnd_scores), dtype=int)
    if len(true_scores)==0 or len(rnd_scores)==0: return float("nan")
    order = np.argsort(-x)                   # descending
    # average ranks for ties
    ranks = np.empty_like(order, dtype=float)
    i=0; r=1.0
    while i < len(order):
        j=i
        while j<len(order) and x[order[j]]==x[order[i]]: j+=1
        avg=(r+(r+(j-i)-1))/2.0
        ranks[order[i:j]] = avg
        r += (j-i); i=j
    rank_sum = ranks[y==1].sum()
    pos = y.sum(); neg = len(y)-pos
    U = rank_sum - pos*(pos+1)/2.0
    return float(U/(pos*neg)) if pos>0 and neg>0 else float("nan")

def evaluate_variant(
    name:str,
    export_df:pd.DataFrame,
    couples_df:pd.DataFrame,
    meta:Dict[str,Any],
    q47_options:Dict[str,str],
    c:float, nmin:int, sym:bool,
    domain_weights:Dict[str,float]|None,
    nonpartner_samples:int=20,
    seed:int=123
):
    rng = random.Random(seed)
    rows = { str(r["participantId"]): r.to_dict() for _,r in export_df.iterrows() }
    ids = list(rows.keys())

    true_scores=[]; rnd_scores=[]
    per_couple=[]

    for _, cpl in couples_df.iterrows():
        A=str(cpl["partner_a_id"]); B=str(cpl["partner_b_id"])
        if A not in rows or B not in rows: continue
        rowA, rowB = rows[A], rows[B]

        prA = extract_priorities_from_q47(rowA, q47_options)
        prB = extract_priorities_from_q47(rowB, q47_options)

        fm, n, sA, sB = final_match(rowA, rowB, meta, prA, prB,
                                    c_penalty=c, min_overlap=nmin, symmetric_check=sym,
                                    domain_weights=domain_weights)
        if fm is None: 
            continue

        true_scores.append(fm)
        per_couple.append({"variant":name,"couple_id":cpl["couple_id"],"score":fm,"overlap":n,"sA":sA,"sB":sB})

        # baseline: random non-partners for A
        distractors=[uid for uid in ids if uid not in (A,B)]
        if len(distractors)==0: continue
        sample_n = min(nonpartner_samples, len(distractors))
        for D in rng.sample(distractors, sample_n):
            rowD = rows[D]
            prD = extract_priorities_from_q47(rowD, q47_options)
            fmAD, _, _, _ = final_match(rowA, rowD, meta, prA, prD,
                                        c_penalty=c, min_overlap=nmin, symmetric_check=sym,
                                        domain_weights=domain_weights)
            if fmAD is not None:
                rnd_scores.append(fmAD)

    # Summary stats
    import numpy as np
    auc  = auc_true_vs_random(true_scores, rnd_scores)
    lift = (float(np.mean(true_scores)) - float(np.mean(rnd_scores))) if (true_scores and rnd_scores) else float("nan")

    # Approximate median rank via sampled distractors
    median_rank = None
    if true_scores and rnd_scores:
        ranks=[]
        for ts in true_scores:
            samp = rng.sample(rnd_scores, min(20, len(rnd_scores)))
            ranks.append(1 + sum(1 for s in samp if s > ts))
        import statistics as st
        median_rank = int(st.median(ranks)) if ranks else None

    return {
        "variant": name,
        "n_couples": len(true_scores),
        "auc": auc,
        "lift": lift,
        "median_rank": median_rank,
        "true_mean": float(np.mean(true_scores)) if true_scores else float("nan"),
        "rnd_mean": float(np.mean(rnd_scores)) if rnd_scores else float("nan"),
    }, per_couple

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_csv", required=True)
    ap.add_argument("--couples_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--questions_json", default=None, help="(Optional) repo data/questions.json (for Q47 mapping)")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--nonpartner_samples", type=int, default=20)
    ap.add_argument("--weights_json", default=None, help="Optional JSON with domain weights to override defaults (keys=Domains)")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    export_df = load_df(args.export_csv)
    couples_df = load_df(args.couples_csv)
    meta = load_meta(args.meta_json)

    # Q47 option labels -> domains
    if args.questions_json:
        q47_options = make_q47_options(args.questions_json)
    else:
        # fallback: assume standard ordering if no file is given
        q47_options = {
            "1":"Shared values", "2":"Communication style", "3":"Activity interests",
            "4":"Daily routine compatibility", "5":"Similar background", "6":"Intellectual connection"
        }

    # optional learned weights
    weights=None
    if args.weights_json:
        with open(args.weights_json,'r',encoding='utf-8') as f:
            w_raw=json.load(f)
        # drop non-domain keys
        cand={k: float(v) for k,v in w_raw.items() if k in DEFAULT_DOMAIN_MULTIPLIER}
        if len(cand)>0:
            # scale to match default sum to keep magnitude comparable
            s_default=sum(DEFAULT_DOMAIN_MULTIPLIER.values())
            s_cand=sum(cand.values()) or 1.0
            scale=s_default/s_cand
            weights={k: v*scale for k,v in cand.items()}

    all_summ=[]; all_rows=[]
    for v in VARIANTS:
        summ, rows = evaluate_variant(
            name=v["name"], export_df=export_df, couples_df=couples_df, meta=meta,
            q47_options=q47_options, c=v["c"], nmin=v["nmin"], sym=v["sym"],
            domain_weights=weights,
            nonpartner_samples=args.nonpartner_samples
        )
        all_summ.append(summ)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(f"{args.outdir}/per_couple_scores.csv", index=False)
    s = pd.DataFrame(all_summ)
    s_sorted = s.sort_values(by=["auc","median_rank","lift"], ascending=[False,True,False])
    s_sorted.to_csv(f"{args.outdir}/summary_by_variant.csv", index=False)

    winner = s_sorted.iloc[0].to_dict()
    with open(f"{args.outdir}/winner.txt","w") as f:
        for k,v in winner.items():
            f.write(f"{k}: {v}\n")

    print("Winner (by AUC, tie-break median rank then lift):")
    print(winner)

if __name__ == "__main__":
    main()
