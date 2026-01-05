from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ensure algorithm/ root on path for sibling packages
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from scoring.config_scorer import ConfigurableScorer, DomainConfig


def _auc_true_vs_random(true_scores: List[float], rnd_scores: List[float]) -> float:
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


def _compute_metrics(
  features_npz: str,
  couples_csv: str,
  config: Dict[str, Any],
  topk: int,
  scores_out: str | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
  scorer = ConfigurableScorer(features_npz)
  cfgs = scorer.make_domain_configs(config)
  couples = pd.read_csv(couples_csv)

  partner_of: Dict[str, str] = {}
  for _, row in couples.iterrows():
    a = str(row.get("partner_a_id"))
    b = str(row.get("partner_b_id"))
    if not a or not b:
      continue
    partner_of[a] = b
    partner_of[b] = a

  anchors = sorted(partner_of.keys())
  true_scores: List[float] = []
  rnd_scores: List[float] = []
  ranks: List[int] = []
  pair_rows: List[Dict[str, Any]] = []
  rank_map: Dict[str, int] = {}

  for aid in anchors:
    partner = partner_of.get(aid)
    if partner is None:
      continue
    # score all candidates for this anchor
    scores: List[Tuple[str, float]] = []
    for bid in scorer.people:
      if bid == aid:
        continue
      s = scorer.score_pair_ids(aid, bid, cfgs)
      scores.append((bid, s))
    if not scores:
      continue
    scores.sort(key=lambda t: t[1], reverse=True)
    # append pair rows for optional export
    for bid, s in scores:
      pair_rows.append(
        {
          "A": aid,
          "B": bid,
          "score": float(s),
          "pct_baseline_like": float("nan"),
        }
      )
    # partner stats
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

  n_anchors = len(rank_map)
  hit1 = float(np.mean([1.0 if r == 0 else 0.0 for r in ranks])) if ranks else float("nan")
  hitk = float(np.mean([1.0 if r < topk else 0.0 for r in ranks])) if ranks else float("nan")
  mrr = float(np.mean([1.0 / (r + 1) for r in ranks])) if ranks else float("nan")
  auc = _auc_true_vs_random(true_scores, rnd_scores)

  # mutual@K over couples
  mutual = 0
  n_couples = 0
  for _, row in couples.iterrows():
    a = str(row.get("partner_a_id"))
    b = str(row.get("partner_b_id"))
    if a not in rank_map or b not in rank_map:
      continue
    n_couples += 1
    if rank_map[a] < topk and rank_map[b] < topk:
      mutual += 1
  mutual_k = float(mutual / n_couples) if n_couples > 0 else float("nan")

  if scores_out and pair_rows:
    out_dir = os.path.dirname(scores_out)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(scores_out, index=False)

  metrics = {
    "n_anchors": float(n_anchors),
    "n_couples": float(n_couples),
    "hit@1": hit1,
    f"hit@{topk}": hitk,
    "mrr": mrr,
    "auc": auc,
    f"mutual@{topk}": mutual_k,
  }
  return metrics, pair_rows


def run_harness(
  features_npz: str,
  couples_csv: str,
  config_json: str,
  outdir: str,
  topk: int = 3,
  scores_basename: str | None = None,
) -> None:
  with open(config_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)
  os.makedirs(outdir, exist_ok=True)

  scores_out = os.path.join(outdir, scores_basename) if scores_basename else None
  base_metrics, _ = _compute_metrics(features_npz, couples_csv, cfg, topk=topk, scores_out=scores_out)

  # domain summary
  summary_rows: List[Dict[str, Any]] = []
  for dom, dcfg in cfg.items():
    summary_rows.append(
      {
        "domain": dom,
        "mode": dcfg.get("mode", "irrelevant"),
        "weight": float(dcfg.get("weight", 0.0)),
        "mu": float(dcfg.get("mu", 0.5)),
        "sigma": float(dcfg.get("sigma", 0.25)),
        "alpha": float(dcfg.get("alpha", 1.0)),
        "beta": float(dcfg.get("beta", 1.0)),
      }
    )
  pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, "domain_summary.csv"), index=False)

  # ablations: drop and flip each domain
  ab_rows: List[Dict[str, Any]] = []
  for dom, dcfg in cfg.items():
    base_hit1 = base_metrics.get("hit@1", float("nan"))
    row: Dict[str, Any] = {
      "domain": dom,
      "mode": dcfg.get("mode", "irrelevant"),
      "weight": float(dcfg.get("weight", 0.0)),
    }
    # drop
    cfg_drop = json.loads(json.dumps(cfg))
    cfg_drop[dom]["weight"] = 0.0
    cfg_drop[dom]["mode"] = "irrelevant"
    drop_metrics, _ = _compute_metrics(features_npz, couples_csv, cfg_drop, topk=topk, scores_out=None)
    row["hit@1_drop"] = drop_metrics.get("hit@1", float("nan"))
    row["d_hit@1_drop"] = row["hit@1_drop"] - base_hit1 if base_hit1 == base_hit1 else float("nan")
    # flip (similarity <-> complementarity)
    cfg_flip = json.loads(json.dumps(cfg))
    mode = str(cfg_flip[dom].get("mode", "irrelevant"))
    if mode == "similarity":
      cfg_flip[dom]["mode"] = "complementarity"
    elif mode == "complementarity":
      cfg_flip[dom]["mode"] = "similarity"
    else:
      cfg_flip[dom]["mode"] = mode
    flip_metrics, _ = _compute_metrics(features_npz, couples_csv, cfg_flip, topk=topk, scores_out=None)
    row["hit@1_flip"] = flip_metrics.get("hit@1", float("nan"))
    row["d_hit@1_flip"] = row["hit@1_flip"] - base_hit1 if base_hit1 == base_hit1 else float("nan")
    ab_rows.append(row)
  pd.DataFrame(ab_rows).to_csv(os.path.join(outdir, "domain_ablation.csv"), index=False)

  # write metrics csv
  with open(os.path.join(outdir, "metrics.csv"), "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    for k, v in base_metrics.items():
      f.write(f"{k},{v}\n")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--features", required=True)
  ap.add_argument("--couples_csv", required=True)
  ap.add_argument("--config", required=True)
  ap.add_argument("--outdir", required=True)
  ap.add_argument("--topk", type=int, default=3)
  ap.add_argument("--scores_basename", default=None, help="optional pair scores csv name (e.g., scores_soft.csv)")
  args = ap.parse_args()
  run_harness(
    features_npz=args.features,
    couples_csv=args.couples_csv,
    config_json=args.config,
    outdir=args.outdir,
    topk=args.topk,
    scores_basename=args.scores_basename,
  )


if __name__ == "__main__":
  main()
