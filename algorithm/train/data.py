from __future__ import annotations

import json, os, sys
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# ensure algorithm/ root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.core import extract_priorities_from_q47
from models.accept_kernels import AcceptabilityParams
from models.domain_weights import DomainWeights
from models.importance_map import ImportanceMap
from models.mc_similarity import build_mc_similarity
from scoring.directional import directional_stats, _z_for_delta


DOMAINS = ["Values","Communication","Lifestyle","Social","Personality","Friendship"]


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


def per_domain_features_pam(
    export_df: pd.DataFrame,
    meta_full: Dict[str, Any],
    A: str,
    B: str,
    delta: float,
    kernels: AcceptabilityParams,
    imp_map: ImportanceMap,
    dom_w_for_feat: DomainWeights,
    sims: Dict[str, Dict[str, Dict[str, float]]],
    q47_options: Dict[str, str],
) -> Tuple[np.ndarray, int]:
    rows = { str(r["participantId"]): r.to_dict() for _, r in export_df.iterrows() }
    if A not in rows or B not in rows:
        return np.zeros(len(DOMAINS), dtype=float), 0
    rowA = rows[A]; rowB = rows[B]
    prA = extract_priorities_from_q47(rowA, q47_options)
    prB = extract_priorities_from_q47(rowB, q47_options)

    feats = []
    total_overlap = 0
    for d in DOMAINS:
        submeta = { k:v for k,v in meta_full.items() if v.get("Domain") == d }
        if not submeta:
            feats.append(0.0)
            continue
        # For features, compute per-domain directional LCBs and combine via geometric mean
        sA, vA, nA, _ = directional_stats(rowA, rowB, submeta, kernels, imp_map, dom_w_for_feat, prA, sims)
        sB, vB, nB, _ = directional_stats(rowB, rowA, submeta, kernels, imp_map, dom_w_for_feat, prB, sims)
        total_overlap += (nA + nB) // 2
        if nA == 0 and nB == 0:
            feats.append(0.0)
            continue
        z = _z_for_delta(delta)
        from math import sqrt
        lA = max(0.0, sA - z * sqrt(max(vA, 0.0)))
        lB = max(0.0, sB - z * sqrt(max(vB, 0.0)))
        feats.append(sqrt(lA * lB))
    return np.array(feats, dtype=float), int(total_overlap)


def build_pairwise_dataset(
    export_df: pd.DataFrame,
    couples_df: pd.DataFrame,
    meta: Dict[str, Any],
    questions_json: str,
    delta: float,
    negatives: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    q47 = make_q47_options(questions_json)
    kernels = AcceptabilityParams.from_defaults(meta)
    imp_map = ImportanceMap.from_defaults()
    dom_w = DomainWeights.from_defaults()
    sims = {}

    rng = np.random.default_rng(seed)
    ids = export_df["participantId"].astype(str).tolist()

    # best-effort sims
    try:
        # dump to a temp CSV for the similarity builder
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tf:
            export_df.to_csv(tf.name, index=False)
            sims = build_mc_similarity(tf.name, questions_json)
    except Exception:
        sims = {}

    X = []
    y = []
    for _, row in couples_df.iterrows():
        A = str(row["partner_a_id"]) ; P = str(row["partner_b_id"])
        if A not in ids or P not in ids:
            continue
        f_pos, _ = per_domain_features_pam(export_df, meta, A, P, delta, kernels, imp_map, dom_w, sims, q47)
        X.append(f_pos)
        y.append(1)
        distractors = [i for i in ids if i not in (A, P)]
        if not distractors:
            continue
        k = min(int(negatives), len(distractors))
        for D in rng.choice(distractors, size=k, replace=False):
            f_neg, _ = per_domain_features_pam(export_df, meta, A, str(D), delta, kernels, imp_map, dom_w, sims, q47)
            X.append(f_neg)
            y.append(0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
