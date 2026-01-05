from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def _parse_json_maybe(s: str):
    if not isinstance(s, str) or s.strip() == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def build_mc_similarity(export_csv_path: str, questions_json_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    returns dict[item_id][o1][o2] -> s in [0,1]
    built via jaccard on acceptable sets across participants
    """
    with open(questions_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    mc_items = {q["id"]: q for q in items if str(q.get("responseType",""))[:2]=="MC"}
    import pandas as pd
    df = pd.read_csv(export_csv_path)

    sims: Dict[str, Dict[str, Dict[str, float]]] = {}
    for qid in mc_items.keys():
        col = f"ACC_{qid}"
        if col not in df.columns:
            continue
        # collect acceptable sets per row, fallback to own selection(s) if empty
        own = df.get(f"SELF_{qid}")
        acc_sets: List[List[str]] = []
        for i, v in enumerate(df[col].tolist()):
            acc = _parse_json_maybe(v)
            opts = []
            if isinstance(acc, dict) and isinstance(acc.get("opts"), list) and acc["opts"]:
                opts = [str(x) for x in acc["opts"]]
            else:
                v0 = own.iloc[i]
                try:
                    t = json.loads(v0) if isinstance(v0, str) else v0
                    if isinstance(t, list):
                        opts = [str(x) for x in t]
                    else:
                        opts = [str(t)]
                except Exception:
                    opts = [str(v0)]
            acc_sets.append(sorted(set(opts)))

        # build co‑accept counts
        co = defaultdict(lambda: defaultdict(int))
        cnt = defaultdict(int)
        for s in acc_sets:
            for o in s:
                cnt[o] += 1
            for i in range(len(s)):
                for j in range(len(s)):
                    co[s[i]][s[j]] += 1

        # jaccard like sim(o,o') = co(o,o')/max(cnt[o]+cnt[o']-co,1)
        M: Dict[str, Dict[str, float]] = {}
        keys = list(cnt.keys())
        for o in keys:
            M[o] = {}
            for o2 in keys:
                inter = co[o][o2]
                union = max(cnt[o] + cnt[o2] - inter, 1)
                s = 1.0 if o == o2 else (inter / union)
                # clamp
                if s < 0.0:
                    s = 0.0
                if s > 1.0:
                    s = 1.0
                M[o][o2] = float(s)
        sims[qid] = M
    return sims

