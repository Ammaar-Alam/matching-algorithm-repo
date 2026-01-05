#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison for Research Paper

Compares all 5 algorithm variants on the same dataset:
1. Evolutionary (learned domain modes via evolutionary search)
2. Soft-Gated (learned domain modes via soft-gating)
3. ML-Weighted (learned domain weights via ML)
4. Merged IDF+LCB (inverse document frequency + lower confidence bounds)
5. Baseline (default equal weights, similarity mode)

Outputs:
- comparison_table.csv: Unified metrics table
- comparison_table.tex: LaTeX-ready table
- domain_weights.csv: Domain weights per algorithm
- per_couple_detail.csv: Per-couple breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ensure algorithm/ root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scoring.config_scorer import ConfigurableScorer


@dataclass
class AlgorithmResult:
    """Results for one algorithm variant."""
    name: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    auc: float
    mutual_at_3: float
    median_rank: float
    mean_rank: float
    domain_weights: Dict[str, float]
    domain_modes: Dict[str, str]
    per_couple_ranks: List[Tuple[str, str, int, int]]  # (a_id, b_id, a_rank_of_b, b_rank_of_a)


def make_baseline_config(domains: List[str]) -> Dict[str, Any]:
    """Create baseline config with equal weights and similarity mode."""
    weight = 1.0 / len(domains)
    return {
        dom: {
            "mode": "similarity",
            "weight": weight,
            "mu": 0.5,
            "sigma": 0.25,
            "alpha": 1.0,
            "beta": 1.0,
        }
        for dom in domains
    }


def auc_true_vs_random(true_scores: List[float], rnd_scores: List[float]) -> float:
    """Compute AUC for true partner scores vs random scores."""
    x = np.array(true_scores + rnd_scores, dtype=float)
    y = np.array([1] * len(true_scores) + [0] * len(rnd_scores), dtype=int)
    if len(true_scores) == 0 or len(rnd_scores) == 0:
        return float("nan")
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    r = 1.0
    while i < len(order):
        j = i
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (r + (r + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg
        r += (j - i)
        i = j
    rank_sum = ranks[y == 1].sum()
    pos = y.sum()
    neg = len(y) - pos
    U = rank_sum - pos * (pos + 1) / 2.0
    return float(U / (pos * neg)) if pos > 0 and neg > 0 else float("nan")


def evaluate_config(
    features_npz: str,
    couples_csv: str,
    config: Dict[str, Any],
    name: str,
) -> AlgorithmResult:
    """Evaluate a single algorithm configuration."""
    scorer = ConfigurableScorer(features_npz)
    cfgs = scorer.make_domain_configs(config)
    couples = pd.read_csv(couples_csv)

    partner_of: Dict[str, str] = {}
    for _, row in couples.iterrows():
        a = str(row.get("partner_a_id"))
        b = str(row.get("partner_b_id"))
        if a and b and a != "nan" and b != "nan":
            partner_of[a] = b
            partner_of[b] = a

    anchors = sorted(partner_of.keys())
    true_scores: List[float] = []
    rnd_scores: List[float] = []
    ranks: List[int] = []
    rank_map: Dict[str, int] = {}
    per_couple_ranks: List[Tuple[str, str, int, int]] = []

    # Compute ranks for each anchor
    for aid in anchors:
        partner = partner_of.get(aid)
        if partner is None:
            continue
        scores: List[Tuple[str, float]] = []
        for bid in scorer.people:
            if bid == aid:
                continue
            s = scorer.score_pair_ids(aid, bid, cfgs)
            scores.append((bid, s))
        if not scores:
            continue
        scores.sort(key=lambda t: t[1], reverse=True)

        pos_score = None
        r = None
        for idx, (bid, s) in enumerate(scores):
            if bid == partner:
                pos_score = float(s)
                r = idx
                break
        if pos_score is None or r is None:
            continue
        true_scores.append(pos_score)
        ranks.append(r)
        rank_map[aid] = r
        for bid, s in scores:
            if bid != partner:
                rnd_scores.append(float(s))

    # Compute per-couple ranks
    seen_couples = set()
    for _, row in couples.iterrows():
        a = str(row.get("partner_a_id"))
        b = str(row.get("partner_b_id"))
        if a in rank_map and b in rank_map:
            key = tuple(sorted([a, b]))
            if key not in seen_couples:
                seen_couples.add(key)
                per_couple_ranks.append((a, b, rank_map[a], rank_map[b]))

    # Compute metrics
    n = len(ranks)
    hit_at_1 = float(np.mean([1.0 if r == 0 else 0.0 for r in ranks])) if ranks else float("nan")
    hit_at_3 = float(np.mean([1.0 if r < 3 else 0.0 for r in ranks])) if ranks else float("nan")
    hit_at_5 = float(np.mean([1.0 if r < 5 else 0.0 for r in ranks])) if ranks else float("nan")
    mrr = float(np.mean([1.0 / (r + 1) for r in ranks])) if ranks else float("nan")
    auc = auc_true_vs_random(true_scores, rnd_scores)
    median_rank = float(np.median([r + 1 for r in ranks])) if ranks else float("nan")
    mean_rank = float(np.mean([r + 1 for r in ranks])) if ranks else float("nan")

    # Mutual@3
    mutual = 0
    n_couples = 0
    for a, b, ra, rb in per_couple_ranks:
        n_couples += 1
        if ra < 3 and rb < 3:
            mutual += 1
    mutual_at_3 = float(mutual / n_couples) if n_couples > 0 else float("nan")

    # Extract domain weights and modes
    domain_weights = {dom: float(cfg.get("weight", 0.0)) for dom, cfg in config.items()}
    domain_modes = {dom: str(cfg.get("mode", "similarity")) for dom, cfg in config.items()}

    return AlgorithmResult(
        name=name,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        mrr=mrr,
        auc=auc,
        mutual_at_3=mutual_at_3,
        median_rank=median_rank,
        mean_rank=mean_rank,
        domain_weights=domain_weights,
        domain_modes=domain_modes,
        per_couple_ranks=per_couple_ranks,
    )


def load_merged_results(summary_ranks_json: str, name: str) -> Optional[AlgorithmResult]:
    """Load results from merged pipeline output (different format)."""
    if not os.path.exists(summary_ranks_json):
        return None

    with open(summary_ranks_json, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    per_person = data.get("per_person", [])

    # Extract per-couple ranks
    per_couple_ranks: List[Tuple[str, str, int, int]] = []
    seen = set()
    for p in per_person:
        anchor = p.get("anchor", "")
        partner = p.get("partner", "")
        rank = p.get("rank", 0)
        key = tuple(sorted([anchor, partner]))
        if key in seen:
            # Find the other rank and complete the tuple
            for i, (a, b, ra, rb) in enumerate(per_couple_ranks):
                if tuple(sorted([a, b])) == key:
                    if a == partner:
                        per_couple_ranks[i] = (a, b, ra, rank)
                    else:
                        per_couple_ranks[i] = (a, b, rank, rb)
                    break
        else:
            seen.add(key)
            per_couple_ranks.append((anchor, partner, rank, 0))

    ranks = [p.get("rank", 0) for p in per_person]
    n = len(ranks)

    return AlgorithmResult(
        name=name,
        hit_at_1=summary.get("hit_at_1", float("nan")),
        hit_at_3=summary.get("hit_at_3", float("nan")),
        hit_at_5=float(np.mean([1.0 if r < 5 else 0.0 for r in ranks])) if ranks else float("nan"),
        mrr=float(np.mean([1.0 / (r + 1) for r in ranks])) if ranks else float("nan"),
        auc=float("nan"),  # Not computed by merged pipeline
        mutual_at_3=float("nan"),  # Would need to compute
        median_rank=summary.get("median_rank", float("nan")),
        mean_rank=summary.get("mean_rank", float("nan")),
        domain_weights={},  # Merged uses IDF weights
        domain_modes={},
        per_couple_ranks=per_couple_ranks,
    )


def generate_latex_table(results: List[AlgorithmResult], outpath: str) -> None:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Algorithm Comparison on Established Couples (n=6)}",
        r"\label{tab:algorithm-comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Algorithm & Hit@1 & Hit@3 & Hit@5 & MRR & AUC & Median Rank \\",
        r"\midrule",
    ]

    for r in results:
        hit1 = f"{r.hit_at_1*100:.0f}\\%" if not np.isnan(r.hit_at_1) else "-"
        hit3 = f"{r.hit_at_3*100:.0f}\\%" if not np.isnan(r.hit_at_3) else "-"
        hit5 = f"{r.hit_at_5*100:.0f}\\%" if not np.isnan(r.hit_at_5) else "-"
        mrr = f"{r.mrr:.2f}" if not np.isnan(r.mrr) else "-"
        auc = f"{r.auc:.2f}" if not np.isnan(r.auc) else "-"
        median = f"{r.median_rank:.1f}" if not np.isnan(r.median_rank) else "-"

        lines.append(f"{r.name} & {hit1} & {hit3} & {hit5} & {mrr} & {auc} & {median} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(outpath, "w") as f:
        f.write("\n".join(lines))


def generate_domain_weights_csv(results: List[AlgorithmResult], outpath: str) -> None:
    """Generate CSV showing domain weights per algorithm."""
    rows = []
    for r in results:
        if not r.domain_weights:
            continue
        for dom, weight in r.domain_weights.items():
            rows.append({
                "algorithm": r.name,
                "domain": dom,
                "weight": weight,
                "mode": r.domain_modes.get(dom, ""),
            })
    if rows:
        pd.DataFrame(rows).to_csv(outpath, index=False)


def generate_per_couple_csv(results: List[AlgorithmResult], names: Dict[str, str], outpath: str) -> None:
    """Generate CSV showing per-couple performance across algorithms."""
    # Collect all couples
    all_couples = set()
    for r in results:
        for a, b, _, _ in r.per_couple_ranks:
            all_couples.add(tuple(sorted([a, b])))

    rows = []
    for couple in sorted(all_couples):
        a, b = couple
        row = {
            "person_a_id": a,
            "person_a_name": names.get(a, a[:8]),
            "person_b_id": b,
            "person_b_name": names.get(b, b[:8]),
        }
        for r in results:
            alg_name = r.name.replace(" ", "_").lower()
            for ca, cb, ra, rb in r.per_couple_ranks:
                if tuple(sorted([ca, cb])) == couple:
                    row[f"{alg_name}_a_rank"] = ra + 1  # 1-indexed
                    row[f"{alg_name}_b_rank"] = rb + 1
                    row[f"{alg_name}_sum_rank"] = ra + rb + 2
                    break
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(outpath, index=False)


def print_comparison(results: List[AlgorithmResult]) -> None:
    """Print human-readable comparison to console."""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    print("\n*** IMPORTANT: Small sample size (n=6 couples) ***")
    print("Each correct prediction changes Hit@1 by ~17%")
    print("Results should be interpreted as preliminary/directional\n")

    # Header
    print(f"{'Algorithm':<20} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'MRR':>8} {'AUC':>8} {'MedRank':>8}")
    print("-" * 70)

    for r in results:
        hit1 = f"{r.hit_at_1*100:.0f}%" if not np.isnan(r.hit_at_1) else "-"
        hit3 = f"{r.hit_at_3*100:.0f}%" if not np.isnan(r.hit_at_3) else "-"
        hit5 = f"{r.hit_at_5*100:.0f}%" if not np.isnan(r.hit_at_5) else "-"
        mrr = f"{r.mrr:.3f}" if not np.isnan(r.mrr) else "-"
        auc = f"{r.auc:.3f}" if not np.isnan(r.auc) else "-"
        median = f"{r.median_rank:.1f}" if not np.isnan(r.median_rank) else "-"
        print(f"{r.name:<20} {hit1:>8} {hit3:>8} {hit5:>8} {mrr:>8} {auc:>8} {median:>8}")

    print("\n" + "-" * 70)
    print("KEY OBSERVATIONS:")

    # Find best per metric
    best_hit1 = max(results, key=lambda x: x.hit_at_1 if not np.isnan(x.hit_at_1) else -1)
    best_auc = max(results, key=lambda x: x.auc if not np.isnan(x.auc) else -1)
    best_median = min(results, key=lambda x: x.median_rank if not np.isnan(x.median_rank) else 999)

    if not np.isnan(best_hit1.hit_at_1):
        print(f"  - Best Hit@1: {best_hit1.name} ({best_hit1.hit_at_1*100:.0f}%)")
    if not np.isnan(best_auc.auc):
        print(f"  - Best AUC: {best_auc.name} ({best_auc.auc:.3f})")
    if not np.isnan(best_median.median_rank):
        print(f"  - Best Median Rank: {best_median.name} ({best_median.median_rank:.1f})")

    print("\nTrade-offs:")
    print("  - Hit@1 = Find the #1 match (best for 'reveals')")
    print("  - AUC = Overall ranking quality (true partners score higher)")
    print("  - Median Rank = Typical partner position in list")
    print("=" * 70)


def load_participant_names(export_csv: str) -> Dict[str, str]:
    """Load participant ID -> name mapping."""
    df = pd.read_csv(export_csv)
    names: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row.get("participantId", ""))
        name = str(row.get("fullName", "")) or str(row.get("netId", "")) or pid[:8]
        if pid and pid != "nan":
            names[pid] = name
    return names


def run_comparison(
    features_npz: str,
    couples_csv: str,
    export_csv: str,
    evo_config: str,
    soft_config: str,
    merged_summary: str,
    outdir: str,
) -> None:
    """Run full algorithm comparison."""
    os.makedirs(outdir, exist_ok=True)

    results: List[AlgorithmResult] = []

    # Load participant names for output
    names = load_participant_names(export_csv)

    # Get domain list from evo config
    with open(evo_config, "r") as f:
        evo_cfg = json.load(f)
    domains = list(evo_cfg.keys())

    # 1. Evolutionary
    print("Evaluating: Evolutionary...")
    results.append(evaluate_config(features_npz, couples_csv, evo_cfg, "Evolutionary"))

    # 2. Soft-Gated
    if os.path.exists(soft_config):
        print("Evaluating: Soft-Gated...")
        with open(soft_config, "r") as f:
            soft_cfg = json.load(f)
        results.append(evaluate_config(features_npz, couples_csv, soft_cfg, "Soft-Gated"))

    # 3. Baseline
    print("Evaluating: Baseline...")
    baseline_cfg = make_baseline_config(domains)
    results.append(evaluate_config(features_npz, couples_csv, baseline_cfg, "Baseline"))

    # 4. Merged IDF+LCB (from pre-computed results)
    if os.path.exists(merged_summary):
        print("Loading: Merged IDF+LCB...")
        merged_result = load_merged_results(merged_summary, "Merged IDF+LCB")
        if merged_result:
            results.append(merged_result)

    # Print comparison
    print_comparison(results)

    # Save outputs
    # Comparison table CSV
    table_rows = []
    for r in results:
        table_rows.append({
            "algorithm": r.name,
            "hit_at_1": r.hit_at_1,
            "hit_at_3": r.hit_at_3,
            "hit_at_5": r.hit_at_5,
            "mrr": r.mrr,
            "auc": r.auc,
            "mutual_at_3": r.mutual_at_3,
            "median_rank": r.median_rank,
            "mean_rank": r.mean_rank,
        })
    pd.DataFrame(table_rows).to_csv(os.path.join(outdir, "comparison_table.csv"), index=False)

    # LaTeX table
    generate_latex_table(results, os.path.join(outdir, "comparison_table.tex"))

    # Domain weights
    generate_domain_weights_csv(results, os.path.join(outdir, "domain_weights.csv"))

    # Per-couple detail
    generate_per_couple_csv(results, names, os.path.join(outdir, "per_couple_detail.csv"))

    # Save baseline config for reference
    with open(os.path.join(outdir, "baseline_config.json"), "w") as f:
        json.dump(baseline_cfg, f, indent=2)

    print(f"\nResults saved to {outdir}/")
    print(f"  - comparison_table.csv")
    print(f"  - comparison_table.tex (LaTeX)")
    print(f"  - domain_weights.csv")
    print(f"  - per_couple_detail.csv")
    print(f"  - baseline_config.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive Algorithm Comparison")
    ap.add_argument("--features", required=True, help="Path to features.npz")
    ap.add_argument("--couples_csv", required=True, help="Path to couples.csv")
    ap.add_argument("--export_csv", required=True, help="Path to export.csv (for names)")
    ap.add_argument("--evo_config", required=True, help="Path to evolutionary best_config.json")
    ap.add_argument("--soft_config", required=True, help="Path to soft-gated best_config.json")
    ap.add_argument("--merged_summary", default="", help="Path to merged pipeline summary_ranks.json")
    ap.add_argument("--outdir", default="out_paper", help="Output directory")
    args = ap.parse_args()

    run_comparison(
        features_npz=args.features,
        couples_csv=args.couples_csv,
        export_csv=args.export_csv,
        evo_config=args.evo_config,
        soft_config=args.soft_config,
        merged_summary=args.merged_summary,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
