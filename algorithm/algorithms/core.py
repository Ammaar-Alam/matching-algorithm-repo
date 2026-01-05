from __future__ import annotations
from math import sqrt
from typing import Dict, Any, Iterable, Tuple

DEFAULT_DOMAIN_MULTIPLIER = {
    "Values": 5.0,
    "Communication": 3.0,
    "Lifestyle": 3.0,
    "Social": 0.5,
    "Personality": 0.5,
    "Friendship": 0.5,
    # Anything else (e.g., IMC/Quality) will be treated as 0.0
}

PRIORITY_LABEL_TO_DOMAIN = {
    # Q47 options -> domains
    "Shared values": "Values",
    "Communication style": "Communication",
    "Activity interests": "Social",
    "Daily routine compatibility": "Lifestyle",
    "Similar background": None,            # No direct domain in the instrument; ignored
    "Intellectual connection": "Friendship"
}

def _importance_int(x) -> int:
    if isinstance(x, int): return x
    if isinstance(x, str):
        x = x.strip()
        if x.isdigit(): return int(x)
        m = {"Irrelevant":0,"A little":1,"Somewhat":10,"Very":50,"Mandatory":250}
        return m.get(x, 1)
    return 1

def _tol_from_label(lbl: str) -> int:
    # Likert tolerance in steps
    m = {"Same answer":0, "±1 step":1, "±2 steps":2, "Any answer":999}
    return m.get((lbl or "±1 step").strip(), 1)

def _is_ok_likert(a_self: str, b_self: str, tol_label: str) -> bool:
    try:
        a = int(a_self); b = int(b_self)
        tol = _tol_from_label(tol_label)
        return abs(b - a) <= tol
    except Exception:
        return False

def _is_ok_mc(b_self: str, opts: Iterable[str]) -> bool:
    return str(b_self) in set(map(str, opts or []))

def _parse_json_maybe(s: str) -> Dict[str, Any]:
    if not isinstance(s, str) or s.strip() == "": return {}
    try:
        import json
        return json.loads(s)
    except Exception:
        return {}

def _domain_multiplier(domain: str, priorities: set[str], base_weights: Dict[str, float] | None = None) -> float:
    weights = base_weights or DEFAULT_DOMAIN_MULTIPLIER
    base = weights.get(domain, 0.0)  # unknown domains (e.g., IMC) -> 0
    if base == 0.0:
        return 0.0
    if domain in priorities:
        return base * 1.5
    return base

def extract_priorities_from_q47(row: Dict[str, Any], q47_options: Dict[str, str]) -> set[str]:
    """
    row: participant row (export.csv row as dict)
    q47_options: map value -> label for Q47 (e.g., "1"->"Shared values")
    Returns the set of domains the user prioritized.
    """
    raw = row.get("SELF_Q47", "")
    domains = set()
    if not raw:
        return domains
    try:
        import json
        vals = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(vals, list): vals = [vals]
        for v in vals:
            lbl = q47_options.get(str(v))
            dom = PRIORITY_LABEL_TO_DOMAIN.get(lbl)
            if dom:
                domains.add(dom)
    except Exception:
        pass
    return domains

def final_match(
    a_row: Dict[str, Any],
    b_row: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
    priorities_a: set[str] | None = None,
    priorities_b: set[str] | None = None,
    c_penalty: float = 100.0,
    min_overlap: int = 20,
    symmetric_check: bool = False,
    domain_weights: Dict[str, float] | None = None,
) -> Tuple[float | None, int, float, float]:
    """
    Compute FinalMatch(A,B) as specified:
      - direction-wise satisfaction with per-item importance and domain multipliers
      - geometric mean for mutuality
      - penalty c/sqrt(n), suppress if n<min_overlap
    Returns: (final_match or None, overlap n, sA, sB)
    """
    priorities_a = priorities_a or set()
    priorities_b = priorities_b or set()

    numA = denA = 0.0
    numB = denB = 0.0
    n = 0

    for qid, m in meta.items():
        # Ignore quality/IMC items by giving them zero base multiplier in meta or unknown domain
        rtype = str(m.get("ResponseType", "Likert5"))
        domain = str(m.get("Domain", ""))

        a_self = a_row.get(f"SELF_{qid}", "")
        b_self = b_row.get(f"SELF_{qid}", "")
        if a_self == "" or b_self == "":
            continue

        # per-user domain-weighted importance
        Wa = _importance_int(a_row.get(f"IMP_{qid}", 1)) * _domain_multiplier(domain, priorities_a, domain_weights)
        Wb = _importance_int(b_row.get(f"IMP_{qid}", 1)) * _domain_multiplier(domain, priorities_b, domain_weights)
        if Wa == 0.0 and Wb == 0.0:
            # counts toward overlap only if it could matter for at least one side
            continue

        # parse acceptable JSON blobs
        accA = _parse_json_maybe(a_row.get(f"ACC_{qid}", ""))
        accB = _parse_json_maybe(b_row.get(f"ACC_{qid}", ""))

        # Evaluate acceptability
        okA = False
        okB = False

        if rtype.startswith("Likert"):
            okA = _is_ok_likert(a_self, b_self, accA.get("tol", "±1 step"))
            okB = _is_ok_likert(b_self, a_self, accB.get("tol", "±1 step"))
        else:
            # MC_SINGLE or MC_MULTI: acceptable is a set; if empty, default to own selection(s)
            def _opts_default(row_val, acc_obj):
                if acc_obj and isinstance(acc_obj.get("opts"), list) and len(acc_obj["opts"]) > 0:
                    return list(map(str, acc_obj["opts"]))
                # default to user's own selection(s)
                try:
                    import json
                    vv = json.loads(row_val) if isinstance(row_val, str) else row_val
                    if isinstance(vv, list):
                        return [str(x) for x in vv]
                    return [str(vv)]
                except Exception:
                    return [str(row_val)]

            def _to_list(v):
                try:
                    import json
                    t = json.loads(v) if isinstance(v, str) else v
                    return t if isinstance(t, list) else [str(v)]
                except Exception:
                    return [str(v)]

            optsA = set(_opts_default(a_self, accA))
            optsB = set(_opts_default(b_self, accB))
            setA = set(_to_list(a_self))
            setB = set(_to_list(b_self))

            if rtype == 'MC_MULTI':
                # MC_MULTI: B acceptable to A if B's set is subset of A's acceptable (or own set if none provided)
                baseA = optsA if len(optsA) > 0 else setA
                baseB = optsB if len(optsB) > 0 else setB
                okA = setB.issubset(baseA)
                okB = setA.issubset(baseB)
            else:
                # MC_SINGLE fallback
                baseA = optsA if len(optsA) > 0 else setA
                baseB = optsB if len(optsB) > 0 else setB
                okA = str(next(iter(setB))) in baseA if setB else False
                okB = str(next(iter(setA))) in baseB if setA else False

        # Only count items that are potentially relevant (any nonzero weight)
        if Wa > 0 or Wb > 0:
            n += 1
            if Wa > 0:
                denA += Wa
                if okA: numA += Wa
            if Wb > 0:
                denB += Wb
                if okB: numB += Wb

    if n < min_overlap:
        return (None, n, 0.0, 0.0)

    sA = 0.0 if denA == 0 else (numA / denA)
    sB = 0.0 if denB == 0 else (numB / denB)

    if symmetric_check:
        match = 100.0 * 0.5 * (sA + sB)  # sensitivity only
    else:
        match = 100.0 * sqrt(sA * sB)    # primary

    penalty = c_penalty / (n ** 0.5)
    final = max(0.0, match - penalty)
    return (final, n, sA, sB)
