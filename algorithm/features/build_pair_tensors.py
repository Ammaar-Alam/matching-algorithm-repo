from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ensure tiger alg root on path so we can reuse existing models
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from algorithms.core import extract_priorities_from_q47
from models.accept_kernels import AcceptabilityParams
from models.importance_map import ImportanceMap
from models.domain_weights import DomainWeights
from scoring.directional import directional_stats, _z_for_delta


DOMAINS = ["Values", "Communication", "Lifestyle", "Social", "Personality", "Friendship"]


def _load_meta(meta_json: str) -> Dict[str, Dict[str, Any]]:
  with open(meta_json, "r", encoding="utf-8") as f:
    return json.load(f)


def _make_q47_options(questions_json_path: str) -> Dict[str, str]:
  with open(questions_json_path, "r", encoding="utf-8") as f:
    items = json.load(f)
  d: Dict[str, str] = {}
  for q in items:
    if q.get("id") == "Q47":
      for idx, label in enumerate(q.get("options", []), start=1):
        d[str(idx)] = label
  return d


def _build_rows(export_csv: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
  df = pd.read_csv(export_csv)
  by_id: Dict[str, Dict[str, Any]] = {}
  for _, r in df.iterrows():
    pid = str(r.get("participantId"))
    if not pid or pid == "nan":
      continue
    by_id[pid] = r.to_dict()
  people = sorted(by_id.keys())
  return by_id, people


def _lcb_from(shat: float, var: float, delta: float) -> float:
  if var <= 1e-12:
    x = max(0.0, min(1.0, float(shat)))
    return x
  z = _z_for_delta(float(delta))
  from math import sqrt
  lo = float(shat) - z * sqrt(float(var))
  if lo < 0.0:
    return 0.0
  if lo > 1.0:
    return 1.0
  return lo


def build_lcbs(export_csv: str, meta_json: str, questions_json: str, delta: float, out_npz: str) -> None:
  by_id, people = _build_rows(export_csv)
  meta = _load_meta(meta_json)
  q47 = _make_q47_options(questions_json)
  # neutral domain weights for per domain lcb features
  dw = DomainWeights(m={d: 1.0 for d in DOMAINS}, psi=1.0)
  kernels = AcceptabilityParams.from_defaults(meta)
  imp = ImportanceMap.from_defaults()

  N = len(people)
  domains = [d for d in DOMAINS if any(m.get("Domain") == d for m in meta.values())]
  D = len(domains)
  lcb = np.zeros((N, N, D, 2), dtype=np.float32)

  # precompute priorities to avoid repeating work
  top3: Dict[str, Any] = {}
  for pid in people:
    row = by_id[pid]
    top3[pid] = extract_priorities_from_q47(row, q47)

  for i, aid in enumerate(people):
    rowA = by_id[aid]
    prA = top3.get(aid) or set()
    for j, bid in enumerate(people):
      if i == j:
        continue
      rowB = by_id[bid]
      prB = top3.get(bid) or set()
      for d_idx, dom in enumerate(domains):
        submeta = {qid: m for qid, m in meta.items() if m.get("Domain") == dom}
        if not submeta:
          continue
        sA, vA, nA, _ = directional_stats(rowA, rowB, submeta, kernels, imp, dw, prA, None)
        sB, vB, nB, _ = directional_stats(rowB, rowA, submeta, kernels, imp, dw, prB, None)
        if nA == 0 and nB == 0:
          continue
        lA = _lcb_from(sA, vA, delta)
        lB = _lcb_from(sB, vB, delta)
        lcb[i, j, d_idx, 0] = lA
        lcb[i, j, d_idx, 1] = lB

  out_dir = os.path.dirname(out_npz)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  np.savez_compressed(out_npz, lcb=lcb, people=np.array(people), domains=np.array(domains))


def build_distances(export_csv: str, meta_json: str, norm_json: str, out_npz: str) -> None:
  df = pd.read_csv(export_csv)
  with open(norm_json, "r", encoding="utf-8") as f:
    norm = json.load(f)
  meta = _load_meta(meta_json)
  by_id, people = _build_rows(export_csv)
  N = len(people)
  domains = [d for d in DOMAINS if any(m.get("Domain") == d for m in meta.values())]
  D = len(domains)

  dist = np.zeros((N, N, D), dtype=np.float32)
  for d_idx, dom in enumerate(domains):
    qids = [qid for qid, m in meta.items() if m.get("Domain") == dom and qid in norm.get("items", {})]
    if not qids:
      continue
    X = np.zeros((N, len(qids)), dtype=np.float32)
    X[:] = np.nan
    for j, qid in enumerate(qids):
      col_name = f"SELF_{qid}"
      if col_name not in df.columns:
        continue
      s = norm["items"].get(qid, {"min": 0.0, "max": 1.0})
      mn = float(s.get("min", 0.0))
      mx = float(s.get("max", 1.0))
      span = mx - mn
      if span <= 0.0:
        span = 1.0
      for i, pid in enumerate(people):
        raw = df.loc[df["participantId"].astype(str) == pid, col_name]
        if raw.empty:
          continue
        val = pd.to_numeric(raw.iloc[0], errors="coerce")
        if not np.isfinite(val):
          continue
        x = (float(val) - mn) / span
        if x < 0.0:
          x = 0.0
        if x > 1.0:
          x = 1.0
        X[i, j] = x
    # compute mean absolute difference per pair ignoring nans
    with np.errstate(invalid="ignore"):
      diff = np.abs(X[:, None, :] - X[None, :, :])
    mask = ~np.isnan(diff)
    num = np.where(mask, diff, 0.0).sum(axis=2)
    cnt = mask.sum(axis=2)
    dom_dist = np.zeros((N, N), dtype=np.float32)
    np.divide(num, np.maximum(cnt, 1), out=dom_dist, where=cnt > 0)
    dist[:, :, d_idx] = dom_dist

  out_dir = os.path.dirname(out_npz)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  np.savez_compressed(out_npz, dist=dist, people=np.array(people), domains=np.array(domains))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["lcbs", "dists"])
  ap.add_argument("--export_csv", required=True)
  ap.add_argument("--meta_json", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--questions_json", default=None, help="data/questions.json path for Q47 mapping")
  ap.add_argument("--delta", type=float, default=0.10)
  ap.add_argument("--norm_json", default=None)
  args = ap.parse_args()
  if args.mode == "lcbs":
    if not args.questions_json:
      raise SystemExit("--questions_json is required for lcbs mode")
    build_lcbs(args.export_csv, args.meta_json, args.questions_json, float(args.delta), args.out)
  else:
    if not args.norm_json:
      raise SystemExit("--norm_json is required for dists mode")
    build_distances(args.export_csv, args.meta_json, args.norm_json, args.out)


if __name__ == "__main__":
  main()

