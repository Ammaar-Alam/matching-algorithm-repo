from __future__ import annotations

import json
from math import sqrt, log
from typing import Dict, Any, Set, Tuple

from algorithms.core import _parse_json_maybe as parse_json
from models.accept_kernels import AcceptabilityParams, p_likert, p_mc
from models.domain_weights import DomainWeights
from models.importance_map import ImportanceMap


def _z_for_delta(delta: float) -> float:
    # common quantiles, small set good enough
    if delta <= 0.05:
        return 1.645
    if delta <= 0.10:
        return 1.2816
    if delta <= 0.20:
        return 0.8416
    return 0.6745


def _to_list(v: Any) -> list:
    try:
        t = json.loads(v) if isinstance(v, str) else v
        if isinstance(t, list):
            return [str(x) for x in t]
        return [str(t)]
    except Exception:
        return [str(v)]


def _likert_tol(acc_obj: Dict[str, Any] | None) -> str:
    if acc_obj and isinstance(acc_obj.get("tol"), str):
        return acc_obj["tol"]
    return "±1 step"


def _acc_set(self_val: Any, acc_obj: Dict[str, Any] | None) -> set[str]:
    if acc_obj and isinstance(acc_obj.get("opts"), list) and acc_obj["opts"]:
        return set(str(x) for x in acc_obj["opts"])
    return set(_to_list(self_val))


def directional_stats(
    rowA: Dict[str, Any],
    rowB: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
    kernels: AcceptabilityParams,
    imp_map: ImportanceMap,
    dom_w: DomainWeights,
    top3A: Set[str] | None,
    mc_sims: Dict[str, Dict[str, Dict[str, float]]] | None,
) -> Tuple[float, float, int, float]:
    """
    returns (s_hat, var_hat, n_items, sumW)
    """
    num = 0.0
    den = 0.0
    vsum = 0.0
    n = 0
    for qid, m in meta.items():
        dom = str(m.get("Domain", ""))
        rt = str(m.get("ResponseType", "Likert5"))
        a_self = rowA.get(f"SELF_{qid}", "")
        b_self = rowB.get(f"SELF_{qid}", "")
        if a_self == "" or b_self == "":
            continue
        W = imp_map.weight(rowA.get(f"IMP_{qid}", 0)) * dom_w.weight(dom, top3A or set())
        if W <= 0.0:
            continue
        p = 0.0
        accA = parse_json(rowA.get(f"ACC_{qid}", ""))
        if rt.startswith("Likert"):
            params = kernels.likert.get(qid)
            if params is None:
                continue
            tol = _likert_tol(accA)
            p = p_likert(params, str(a_self), str(b_self), tol)
        else:
            params = kernels.mc.get(qid)
            if params is None:
                continue
            Sa = _acc_set(a_self, accA)
            sims = mc_sims.get(qid, {}) if mc_sims else {}
            # get similarity of b's answer to a's accepted set
            best = 0.0
            for y in _to_list(b_self):
                for o in Sa:
                    s = sims.get(str(y), {}).get(str(o), 1.0 if y == o else 0.0)
                    if s > best:
                        best = s
            p = p_mc(params, best)

        # accumulate
        n += 1
        den += W
        num += W * p
        vsum += (W * W) * (p * (1.0 - p))

    if den <= 0.0 or n == 0:
        return 0.0, 0.0, 0, 0.0
    shat = num / den
    var = vsum / (den * den) if den > 0 else 0.0
    return float(shat), float(var), int(n), float(den)

