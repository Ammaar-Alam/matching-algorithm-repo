#!/usr/bin/env python3
"""
Production Matching Pipeline

Generates:
1. Preference lists (top-K candidates per person)
2. Optimal 1:1 matching (max-weight matching via Blossom algorithm)
3. Mutual matches (pairs where both rank each other in top-K)

Supports multiple scoring algorithms:
- Evolutionary (learned domain modes)
- Soft-gated (learned domain modes)
- ML-weighted (learned domain weights)
- Baseline (default domain weights)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

# ensure algorithm/ root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scoring.config_scorer import ConfigurableScorer


@dataclass
class CandidateMatch:
    """A single candidate in someone's preference list."""
    rank: int
    candidate_id: str
    candidate_name: str
    score: float
    pct: float  # percentage (0-100)
    is_mutual: bool


@dataclass
class PersonPreferences:
    """Preference list for one person."""
    person_id: str
    person_name: str
    preferences: List[CandidateMatch]


@dataclass
class MatchPair:
    """A matched pair in 1:1 matching."""
    person_a_id: str
    person_a_name: str
    person_b_id: str
    person_b_name: str
    score: float
    pct: float
    mutual_ranks: Tuple[int, int]  # (a's rank of b, b's rank of a)


@dataclass
class MutualMatch:
    """A mutual match where both rank each other in top-K."""
    person_a_id: str
    person_a_name: str
    person_b_id: str
    person_b_name: str
    a_ranks_b: int
    b_ranks_a: int
    score: float


def load_participant_names(export_csv: str) -> Dict[str, str]:
    """Load participant ID -> name mapping from export."""
    df = pd.read_csv(export_csv)
    names: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row.get("participantId", ""))
        name = str(row.get("fullName", "")) or str(row.get("netId", "")) or pid[:8]
        if pid and pid != "nan":
            names[pid] = name
    return names


def compute_all_scores(
    features_npz: str,
    config: Dict[str, Any],
    names: Dict[str, str],
) -> Tuple[Dict[str, List[Tuple[str, float]]], float, float]:
    """
    Compute scores for all pairs using the configurable scorer.

    Returns:
        - scores_by_person: {person_id: [(other_id, score), ...] sorted by score desc}
        - min_score: minimum score seen
        - max_score: maximum score seen
    """
    scorer = ConfigurableScorer(features_npz)
    cfgs = scorer.make_domain_configs(config)

    people = list(scorer.people)
    scores_by_person: Dict[str, List[Tuple[str, float]]] = {p: [] for p in people}
    all_scores: List[float] = []

    for i, pid_a in enumerate(people):
        for pid_b in people:
            if pid_a == pid_b:
                continue
            score = scorer.score_pair_ids(pid_a, pid_b, cfgs)
            scores_by_person[pid_a].append((pid_b, score))
            all_scores.append(score)

    # Sort each person's candidates by score descending
    for pid in people:
        scores_by_person[pid].sort(key=lambda x: x[1], reverse=True)

    min_score = min(all_scores) if all_scores else 0.0
    max_score = max(all_scores) if all_scores else 1.0

    return scores_by_person, min_score, max_score


def score_to_pct(score: float, min_score: float, max_score: float) -> float:
    """Convert raw score to percentage (0-100)."""
    if max_score <= min_score:
        return 50.0
    return 100.0 * (score - min_score) / (max_score - min_score)


def build_preference_lists(
    scores_by_person: Dict[str, List[Tuple[str, float]]],
    names: Dict[str, str],
    min_score: float,
    max_score: float,
    top_k: int,
) -> List[PersonPreferences]:
    """Build top-K preference lists for each person."""

    # First, compute reverse rankings for mutual detection
    rank_of: Dict[str, Dict[str, int]] = {}  # rank_of[a][b] = a's rank of b
    for pid, candidates in scores_by_person.items():
        rank_of[pid] = {}
        for rank, (cid, _) in enumerate(candidates):
            rank_of[pid][cid] = rank

    results: List[PersonPreferences] = []

    for pid, candidates in scores_by_person.items():
        prefs: List[CandidateMatch] = []
        for rank, (cid, score) in enumerate(candidates[:top_k]):
            # Check if mutual (both rank each other in top-K)
            other_rank = rank_of.get(cid, {}).get(pid, float("inf"))
            is_mutual = other_rank < top_k

            prefs.append(CandidateMatch(
                rank=rank + 1,
                candidate_id=cid,
                candidate_name=names.get(cid, cid[:8]),
                score=score,
                pct=score_to_pct(score, min_score, max_score),
                is_mutual=is_mutual,
            ))

        results.append(PersonPreferences(
            person_id=pid,
            person_name=names.get(pid, pid[:8]),
            preferences=prefs,
        ))

    # Sort by highest top score
    results.sort(key=lambda x: x.preferences[0].score if x.preferences else 0, reverse=True)

    return results


def compute_optimal_matching(
    scores_by_person: Dict[str, List[Tuple[str, float]]],
    names: Dict[str, str],
    min_score: float,
    max_score: float,
) -> List[MatchPair]:
    """Compute optimal 1:1 matching using max-weight matching."""
    from matching.blossom import max_weight_matching_exact

    # Build symmetric weights (average of directional scores)
    people = list(scores_by_person.keys())
    weights: Dict[Tuple[str, str], float] = {}

    # Create lookup for fast access
    score_lookup: Dict[Tuple[str, str], float] = {}
    for pid, candidates in scores_by_person.items():
        for cid, score in candidates:
            score_lookup[(pid, cid)] = score

    for i, a in enumerate(people):
        for b in people[i+1:]:
            s_ab = score_lookup.get((a, b), 0.0)
            s_ba = score_lookup.get((b, a), 0.0)
            weights[(a, b)] = (s_ab + s_ba) / 2.0

    matching = max_weight_matching_exact(people, weights)

    # Build rank lookup
    rank_of: Dict[str, Dict[str, int]] = {}
    for pid, candidates in scores_by_person.items():
        rank_of[pid] = {cid: rank for rank, (cid, _) in enumerate(candidates)}

    # Convert to MatchPair objects (avoiding duplicates)
    seen: set = set()
    pairs: List[MatchPair] = []
    for a, b in matching.items():
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)

        score = weights.get((min(a, b), max(a, b)), 0.0)
        a_rank_b = rank_of.get(a, {}).get(b, -1) + 1
        b_rank_a = rank_of.get(b, {}).get(a, -1) + 1

        pairs.append(MatchPair(
            person_a_id=a,
            person_a_name=names.get(a, a[:8]),
            person_b_id=b,
            person_b_name=names.get(b, b[:8]),
            score=score,
            pct=score_to_pct(score, min_score, max_score),
            mutual_ranks=(a_rank_b, b_rank_a),
        ))

    pairs.sort(key=lambda x: x.score, reverse=True)
    return pairs


def find_mutual_matches(
    scores_by_person: Dict[str, List[Tuple[str, float]]],
    names: Dict[str, str],
    top_k: int,
) -> List[MutualMatch]:
    """Find pairs where both rank each other in top-K."""

    # Build rank lookup
    rank_of: Dict[str, Dict[str, int]] = {}
    for pid, candidates in scores_by_person.items():
        rank_of[pid] = {cid: rank + 1 for rank, (cid, _) in enumerate(candidates)}

    # Build score lookup
    score_lookup: Dict[Tuple[str, str], float] = {}
    for pid, candidates in scores_by_person.items():
        for cid, score in candidates:
            score_lookup[(pid, cid)] = score

    mutual: List[MutualMatch] = []
    seen: set = set()

    for pid, candidates in scores_by_person.items():
        for cid, _ in candidates[:top_k]:
            key = tuple(sorted([pid, cid]))
            if key in seen:
                continue

            # Check if cid also ranks pid in top-K
            cid_rank_of_pid = rank_of.get(cid, {}).get(pid, float("inf"))
            if cid_rank_of_pid <= top_k:
                seen.add(key)
                s_ab = score_lookup.get((pid, cid), 0.0)
                s_ba = score_lookup.get((cid, pid), 0.0)

                mutual.append(MutualMatch(
                    person_a_id=pid,
                    person_a_name=names.get(pid, pid[:8]),
                    person_b_id=cid,
                    person_b_name=names.get(cid, cid[:8]),
                    a_ranks_b=rank_of.get(pid, {}).get(cid, -1),
                    b_ranks_a=int(cid_rank_of_pid),
                    score=(s_ab + s_ba) / 2.0,
                ))

    mutual.sort(key=lambda x: (x.a_ranks_b + x.b_ranks_a, -x.score))
    return mutual


def print_results(
    preference_lists: List[PersonPreferences],
    optimal_matching: List[MatchPair],
    mutual_matches: List[MutualMatch],
    algorithm_name: str,
    top_k: int,
) -> None:
    """Print human-readable results to console."""

    print("\n" + "=" * 60)
    print(f"=== Production Matching Results ===")
    print(f"Algorithm: {algorithm_name}")
    print("=" * 60)

    # Preference Lists
    print(f"\nPREFERENCE LISTS (Top {top_k} per person):")
    print("-" * 50)

    for person in preference_lists:
        print(f"\n{person.person_name}")
        for match in person.preferences:
            mutual_marker = " [mutual]" if match.is_mutual else ""
            print(f"   #{match.rank}: {match.candidate_name} ({match.pct:.1f}%){mutual_marker}")

    # Optimal 1:1 Matching
    print(f"\n\nOPTIMAL 1:1 MATCHING ({len(optimal_matching)} pairs):")
    print("-" * 50)

    for pair in optimal_matching:
        rank_info = f"[ranks: {pair.mutual_ranks[0]}, {pair.mutual_ranks[1]}]"
        print(f"{pair.person_a_name} <-> {pair.person_b_name}  ({pair.pct:.1f}%)  {rank_info}")

    # Mutual Matches
    print(f"\n\nMUTUAL TOP-{top_k} MATCHES ({len(mutual_matches)} pairs):")
    print("-" * 50)

    if mutual_matches:
        for match in mutual_matches:
            print(f"{match.person_a_name} <-> {match.person_b_name}  [mutual #{match.a_ranks_b}, #{match.b_ranks_a}]")
    else:
        print("(no mutual matches in top-K)")

    print("\n" + "=" * 60)


def save_results(
    preference_lists: List[PersonPreferences],
    optimal_matching: List[MatchPair],
    mutual_matches: List[MutualMatch],
    outdir: str,
) -> None:
    """Save results to JSON and CSV files."""
    os.makedirs(outdir, exist_ok=True)

    # Preference lists JSON
    prefs_data = []
    for person in preference_lists:
        prefs_data.append({
            "person_id": person.person_id,
            "person_name": person.person_name,
            "preferences": [asdict(m) for m in person.preferences],
        })
    with open(os.path.join(outdir, "preference_lists.json"), "w") as f:
        json.dump(prefs_data, f, indent=2)

    # Optimal matching JSON
    matching_data = [asdict(p) for p in optimal_matching]
    with open(os.path.join(outdir, "optimal_matching.json"), "w") as f:
        json.dump(matching_data, f, indent=2)

    # Mutual matches JSON
    mutual_data = [asdict(m) for m in mutual_matches]
    with open(os.path.join(outdir, "mutual_matches.json"), "w") as f:
        json.dump(mutual_data, f, indent=2)

    # Summary CSV
    summary_rows = []
    for person in preference_lists:
        for match in person.preferences:
            summary_rows.append({
                "person_id": person.person_id,
                "person_name": person.person_name,
                "rank": match.rank,
                "candidate_id": match.candidate_id,
                "candidate_name": match.candidate_name,
                "score": match.score,
                "pct": match.pct,
                "is_mutual": match.is_mutual,
            })
    pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, "preference_lists.csv"), index=False)

    # Matching CSV
    pd.DataFrame([asdict(p) for p in optimal_matching]).to_csv(
        os.path.join(outdir, "optimal_matching.csv"), index=False
    )

    print(f"\nResults saved to {outdir}/")
    print(f"  - preference_lists.json")
    print(f"  - preference_lists.csv")
    print(f"  - optimal_matching.json")
    print(f"  - optimal_matching.csv")
    print(f"  - mutual_matches.json")


def run_production_matching(
    features_npz: str,
    export_csv: str,
    config_json: str,
    outdir: str,
    top_k: int = 5,
    algorithm_name: str = "Unknown",
) -> None:
    """Main entry point for production matching."""

    # Load config
    with open(config_json, "r") as f:
        config = json.load(f)

    # Load names
    names = load_participant_names(export_csv)

    # Compute all scores
    print("Computing pair scores...")
    scores_by_person, min_score, max_score = compute_all_scores(features_npz, config, names)

    # Build preference lists
    print("Building preference lists...")
    preference_lists = build_preference_lists(scores_by_person, names, min_score, max_score, top_k)

    # Compute optimal matching
    print("Computing optimal 1:1 matching...")
    optimal_matching = compute_optimal_matching(scores_by_person, names, min_score, max_score)

    # Find mutual matches
    print("Finding mutual matches...")
    mutual_matches = find_mutual_matches(scores_by_person, names, top_k)

    # Print results
    print_results(preference_lists, optimal_matching, mutual_matches, algorithm_name, top_k)

    # Save results
    save_results(preference_lists, optimal_matching, mutual_matches, outdir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Production Matching Pipeline")
    ap.add_argument("--features", required=True, help="Path to features.npz")
    ap.add_argument("--export_csv", required=True, help="Path to export.csv")
    ap.add_argument("--config", required=True, help="Path to config JSON (e.g., out_evo/best_config.json)")
    ap.add_argument("--outdir", default="out_production", help="Output directory")
    ap.add_argument("--top_k", type=int, default=5, help="Number of top candidates per person")
    ap.add_argument("--algorithm", default="Unknown", help="Algorithm name for display")
    args = ap.parse_args()

    run_production_matching(
        features_npz=args.features,
        export_csv=args.export_csv,
        config_json=args.config,
        outdir=args.outdir,
        top_k=args.top_k,
        algorithm_name=args.algorithm,
    )


if __name__ == "__main__":
    main()
