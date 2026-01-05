#!/usr/bin/env python3
"""
Print merged matching results with participant names instead of IDs.
"""
from pathlib import Path
import csv
import json
import re
import sys

ROOT = Path(".")
MERGED = ROOT / "tiger-alg" / "out_merged"

def load_id2name():
    """Load participant ID -> name mapping from export.csv and couples.csv"""
    id2name = {}

    def cleaned(s):
        return " ".join(str(s).strip().split())

    def best_name(row):
        # try common name columns (prioritize fullName which is in export.csv)
        for c in ["fullName", "displayName", "name", "full_name", "participant_name", "preferred_name", "username"]:
            if c in row and row[c]:
                val = cleaned(row[c])
                if val:
                    return val
        # try first/last
        first = ""
        for c in ["first_name", "first", "given_name"]:
            if c in row and row[c]:
                first = cleaned(row[c])
                break
        last = ""
        for c in ["last_name", "last", "surname", "family_name"]:
            if c in row and row[c]:
                last = cleaned(row[c])
                break
        nm = (first + " " + last).strip()
        return nm if nm else None

    # 1) export.csv - this is the primary source
    exp = ROOT / "export.csv"
    if exp.exists():
        with exp.open(newline="") as f:
            r = csv.DictReader(f)
            # Look for participantId column (case-insensitive)
            id_cols = [c for c in r.fieldnames if re.search(r'(participant.*id|participantId|^id$|^uuid$)', c, re.I)]
            if not id_cols:
                print(f"Warning: No participant ID column found in export.csv. Available columns: {list(r.fieldnames)[:10]}", file=sys.stderr)
            for row in r:
                pid = None
                for col in id_cols:
                    if row.get(col):
                        pid = str(row[col]).strip()
                        break
                if not pid:
                    continue
                nm = best_name(row)
                if nm:
                    id2name[pid] = nm
                    # Also store shortened version (first 8 chars) for compatibility
                    if len(pid) > 8:
                        short_id = pid[:8]
                        if short_id not in id2name:  # Don't overwrite if full ID exists
                            id2name[short_id] = nm

    # 2) couples.csv (partner_a_id/name, partner_b_id/name, or any partner*id/name pair)
    coup = ROOT / "couples.csv"
    if coup.exists():
        with coup.open(newline="") as f:
            r = csv.DictReader(f)
            fns = r.fieldnames or []
            # map partner prefixes (e.g., partner_a_, partner_b_, a_, b_)
            prefixes = set()
            for c in fns:
                m = re.match(r'(.+?)(id|name)$', c, flags=re.I)
                if m:
                    prefixes.add(m.group(1))
            for row in r:
                for pre in prefixes:
                    pid = row.get(pre + "id") or row.get(pre + "ID") or row.get(pre + "Id")
                    nm = row.get(pre + "name") or row.get(pre + "Name")
                    if pid and nm:
                        pid = str(pid).strip()
                        id2name[pid] = cleaned(nm)
                        # Also store shortened version
                        if len(pid) > 8:
                            short_id = pid[:8]
                            if short_id not in id2name:
                                id2name[short_id] = cleaned(nm)

    # 3) pair_scores_named.csv (if you generated it) - but only use if it has actual names, not IDs
    psn = MERGED / "pair_scores_named.csv"
    if psn.exists():
        with psn.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                a, an = row.get("A"), row.get("A_name")
                b, bn = row.get("B"), row.get("B_name")
                # Only use if the "name" field doesn't look like an ID (8 hex chars)
                def looks_like_id(s):
                    s = str(s).strip()
                    return len(s) == 8 and all(c in '0123456789abcdef' for c in s.lower())
                
                if a and an and not looks_like_id(an):
                    # Only update if we don't already have a better mapping
                    a_str = str(a).strip()
                    if a_str not in id2name:  # Don't overwrite existing mappings
                        id2name[a_str] = cleaned(an)
                        if len(a_str) > 8:
                            short_id = a_str[:8]
                            if short_id not in id2name:
                                id2name[short_id] = cleaned(an)
                if b and bn and not looks_like_id(bn):
                    b_str = str(b).strip()
                    if b_str not in id2name:  # Don't overwrite existing mappings
                        id2name[b_str] = cleaned(bn)
                        if len(b_str) > 8:
                            short_id = b_str[:8]
                            if short_id not in id2name:
                                id2name[short_id] = cleaned(bn)
    
    return id2name

id2name = load_id2name()
if not id2name:
    print("WARNING: No names loaded! Check export.csv", file=sys.stderr)

# create stable placeholders for any remaining unknown ids (no raw IDs shown)
def build_placeholders(all_ids, have):
    unknown = sorted([x for x in all_ids if x not in have])
    return {pid: f"Participant {i+1}" for i, pid in enumerate(unknown)}

def nm(pid, id2name_dict, ph):
    """Get name for participant ID, trying full ID, shortened ID, then placeholder"""
    if not pid:
        return "Participant"
    pid_str = str(pid).strip()
    # Try full ID first
    if pid_str in id2name_dict:
        return id2name_dict[pid_str]
    # Try shortened version (first 8 chars)
    if len(pid_str) > 8:
        short_id = pid_str[:8]
        if short_id in id2name_dict:
            return id2name_dict[short_id]
    # Try placeholder
    if pid_str in ph:
        return ph[pid_str]
    # Last resort: try to find by prefix match
    for known_id, name in id2name_dict.items():
        if pid_str.startswith(known_id[:8]) or known_id.startswith(pid_str[:8]):
            return name
    return "Participant"

# ---------- load merged artifacts ----------
pairs_csv = MERGED / "pair_scores_merged.csv"
summary_json = MERGED / "summary_ranks.json"
da_json = MERGED / "matching_da.json"
lp_json = MERGED / "matching_best_stable_lp.json"
mw_json = MERGED / "matching_max_weight.json"

if not pairs_csv.exists() or not summary_json.exists():
    print("Run the merged pipeline first. Expected:", pairs_csv, "and", summary_json, file=sys.stderr)
    sys.exit(1)

# directional LCB and symmetric HM_mutual
LCB, HMmut, ALL_IDS = {}, {}, set()
rows = list(csv.DictReader(pairs_csv.read_text().splitlines()))
for r in rows:
    a, b = r["A"], r["B"]
    la = float(r.get("LCB_A", 0.0) or 0.0)
    lb = float(r.get("LCB_B", 0.0) or 0.0)
    hm = float(r.get("HM_mutual", r.get("HM", 0.0)) or 0.0)
    LCB.setdefault(a, {})[b] = la
    LCB.setdefault(b, {})[a] = lb
    HMmut.setdefault(a, {})[b] = hm
    HMmut.setdefault(b, {})[a] = hm
    ALL_IDS.add(a)
    ALL_IDS.add(b)

# placeholders so we never print bare IDs
placeholders = build_placeholders(ALL_IDS, id2name)

# strict lists by directional LCB
lists = {a: [b for b, _ in sorted(LCB[a].items(), key=lambda t: t[1], reverse=True)] for a in LCB}

def partner_of(m, a):
    if a in m:
        return m[a]
    for x, y in m.items():
        if y == a:
            return x
    return None

def blocking_pairs_named(m):
    blks = set()
    ids = list(lists.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ma, mb = partner_of(m, a), partner_of(m, b)
            La, Lb = lists.get(a, []), lists.get(b, [])
            better_a = (b in La) and (ma is None or (ma not in La or La.index(b) < La.index(ma)))
            better_b = (a in Lb) and (mb is None or (mb not in Lb or Lb.index(a) < Lb.index(mb)))
            if better_a and better_b:
                blks.add((nm(a, id2name, placeholders), nm(b, id2name, placeholders)))
    return sorted(blks)

def matching_edges_named(m):
    seen, out = set(), []
    for a, b in m.items():
        u, v = sorted((a, b))
        if (u, v) in seen:
            continue
        seen.add((u, v))
        u_name = nm(u, id2name, placeholders)
        v_name = nm(v, id2name, placeholders)
        out.append((u_name, v_name))
    return sorted(out)

# ---------- print: per person (directional), per‑algo, matchings, diagnostics ----------
summ = json.loads(summary_json.read_text())
print("\n=== MERGED (directional LCB) — per participant ===")
S = summ.get("summary", {})
print(f"Hit@1={S.get('hit_at_1',0):.2f}  Hit@3={S.get('hit_at_3',0):.2f}  median={S.get('median_rank',0):.2f}  n={S.get('n_people',0)}\n")

for r in summ.get("per_person", []):
    a, p = r["anchor"], r["partner"]
    aN, pN = nm(a, id2name, placeholders), nm(p, id2name, placeholders)
    rr = r.get("rank")
    gap = r.get("top_gap", float("nan"))
    tag = r.get("robustness", "?")
    cands = sorted(HMmut.get(a, {}).items(), key=lambda t: t[1], reverse=True)[:3]
    top3 = " | ".join(f"{nm(b, id2name, placeholders)}({s:.3f})" for b, s in cands)
    print(f"  {aN} → true partner: {pN}")
    print(f"    true-rank={rr}  gap={gap:.3f}  ({tag})")
    print(f"    top3: {top3}\n")

# Baseline/ML/PAM/Soft/Evo per-person comparison if available
alg_csv = ROOT / "analysis" / "alg_compare.csv"
if alg_csv.exists():
    print("\n=== BASELINES vs ML vs PAM vs SOFT vs EVO — per participant ===")
    rows = list(csv.DictReader(alg_csv.read_text().splitlines()))
    for row in rows:
        a, p = row["anchor_id"], row["partner_id"]
        aN, pN = nm(a, id2name, placeholders), nm(p, id2name, placeholders)
        
        print(f"\n  {aN} (true partner: {pN})")
        
        def format_algo(prefix):
            top_id = row.get(f"{prefix}_top_id", "")
            top_name = nm(top_id, id2name, placeholders) if top_id else "(none)"
            pr = row.get(f"{prefix}_partner_rank", "") or "—"
            score = row.get(f"{prefix}_top_score", "")
            score_str = f"  score={float(score):.3f}" if score else ""
            return f"    {prefix.upper():8s}: top={top_name:20s}  rank={pr:2s}{score_str}"
        
        for algo in ["baseline", "ml", "pam", "soft", "evo"]:
            print(format_algo(algo))
else:
    print("\n(no analysis/alg_compare.csv found — run your baseline/ML/PAM/soft/evo evaluations to populate it)")

# Matchings & blocking pairs
DA = json.loads(da_json.read_text()) if da_json.exists() else {}
LP = json.loads(lp_json.read_text()) if lp_json.exists() else {}
MW = json.loads(mw_json.read_text()) if mw_json.exists() else {}

print("\n=== MATCHING COMPARISON (names) ===")
if DA:
    matches = matching_edges_named(DA)
    print(f"  DA        : {matches}")
if LP:
    matches = matching_edges_named(LP)
    print(f"  BestStable: {matches}")
if MW:
    matches = matching_edges_named(MW)
    print(f"  MaxWeight : {matches}")

if lists:
    print()
    if DA:
        bp = blocking_pairs_named(DA)
        bp_str = ", ".join(f"({a}, {b})" for a, b in bp[:8])
        if len(bp) > 8:
            bp_str += " …"
        print(f"  Blocking pairs (DA): {len(bp)}  →  {bp_str}")
    if LP:
        bp = blocking_pairs_named(LP)
        bp_str = ", ".join(f"({a}, {b})" for a, b in bp[:8])
        if len(bp) > 8:
            bp_str += " …"
        print(f"  Blocking pairs (BestStable): {len(bp)}  →  {bp_str}")
    if MW:
        bp = blocking_pairs_named(MW)
        bp_str = ", ".join(f"({a}, {b})" for a, b in bp[:8])
        if len(bp) > 8:
            bp_str += " …"
        print(f"  Blocking pairs (MaxWeight): {len(bp)}  →  {bp_str}")

# Fragile anchors: mutual‑top‑2 fallbacks (directional lists)
print("\n=== Fragile anchors — mutual‑top‑2 fallback suggestions ===")
ranks = {a: {b: i + 1 for i, b in enumerate(lists.get(a, []))} for a in lists}
fragile_found = False
for r in summ.get("per_person", []):
    if r.get("robustness") != "fragile":
        continue
    fragile_found = True
    a, p = r["anchor"], r["partner"]
    aN, pN = nm(a, id2name, placeholders), nm(p, id2name, placeholders)
    ra = ranks.get(a, {}).get(p)
    mutual2 = []
    for b in lists.get(a, [])[:5]:
        ra1 = ranks.get(a, {}).get(b, 10**9)
        rb1 = ranks.get(b, {}).get(a, 10**9)
        if ra1 <= 2 and rb1 <= 2:
            mutual2.append(f"{nm(b, id2name, placeholders)}(a→{ra1}, b→{rb1})")
    print(f"\n  {aN} (true partner: {pN}, rank={ra})")
    if mutual2:
        print(f"    alternatives: {'; '.join(mutual2)}")
    else:
        print(f"    alternatives: —")
if not fragile_found:
    print("  (none)")

print("\nDone.\n")

