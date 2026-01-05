from __future__ import annotations

import argparse, os, sys, json
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ensure algorithm/ root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.data import load_meta, build_pairwise_dataset, DOMAINS
from models.domain_weights import DomainWeights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_csv", required=True)
    ap.add_argument("--couples_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--questions_json", required=True)
    ap.add_argument("--outdir", default="out_pam")
    ap.add_argument("--negatives", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    export_df = pd.read_csv(args.export_csv)
    couples_df = pd.read_csv(args.couples_csv)
    meta = load_meta(args.meta_json)

    X, y = build_pairwise_dataset(export_df, couples_df, meta, args.questions_json, args.delta, args.negatives, seed=args.seed)
    if len(X) == 0:
        # tiny-data fallback: write defaults and exit gracefully
        defaults = DomainWeights.from_defaults().m
        out = { **{k: float(v) for k,v in defaults.items()}, "cv_auc": float('nan'), "note": "fallback_defaults_no_data" }
        with open(os.path.join(args.outdir, "weights.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return

    if args.verbose:
        pos = int((y==1).sum()); neg = int((y==0).sum())
        print(f"dataset: n={len(y)} (pos={pos}, neg={neg}), features={X.shape[1]}")

    # stratified K-fold CV for AUC (handle single-class gracefully)
    pos = int((y==1).sum()); neg = int((y==0).sum())
    if pos == 0 or neg == 0:
        defaults = DomainWeights.from_defaults().m
        out = { **{k: float(v) for k,v in defaults.items()}, "cv_auc": float('nan'), "note": "fallback_defaults_single_class" }
        with open(os.path.join(args.outdir, "weights.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return
    coef = None; auc = float('nan')
    if pos >= 2 and neg >= 2:
        n_splits = max(2, min(5, pos, neg))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        aucs = []; coefs = []
        for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
            clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
            coefs.append(clf.coef_[0])
            if args.verbose:
                print(f"fold {fold}/{n_splits} AUC={aucs[-1]:.3f}")
        coef = np.mean(coefs, axis=0)
        auc = float(np.mean(aucs))
    else:
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
        clf.fit(X, y)
        coef = clf.coef_[0]

    eps = 1e-3
    pos_coef = np.maximum(coef, eps)
    # normalize to align magnitude with defaults
    default_sum = sum(DomainWeights.from_defaults().m.values())
    norm = pos_coef / (pos_coef.sum() if pos_coef.sum() > 0 else 1.0)
    learned = { d: float(default_sum * norm[i]) for i, d in enumerate(DOMAINS) }
    learned["cv_auc"] = auc

    with open(os.path.join(args.outdir, "weights.json"), "w") as f:
        json.dump(learned, f, indent=2)
    if args.verbose:
        print("learned domain weights:")
        for d in DOMAINS:
            print(f"  {d:12s} {learned[d]: .6f}")
        print(f"cv_auc={auc}")
    print(json.dumps(learned, indent=2))


if __name__ == "__main__":
    main()
