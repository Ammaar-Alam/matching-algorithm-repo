from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd


DOMAINS = ["Values", "Communication", "Lifestyle", "Social", "Personality", "Friendship"]


def _load_meta(meta_json: str) -> Dict[str, Dict[str, Any]]:
  with open(meta_json, "r", encoding="utf-8") as f:
    return json.load(f)


def fit_minmax(export_csv: str, meta_json: str, out_json: str) -> Dict[str, Dict[str, float]]:
  df = pd.read_csv(export_csv)
  meta = _load_meta(meta_json)
  # use only pam domains
  item_ids = [qid for qid, m in meta.items() if str(m.get("Domain")) in DOMAINS]
  stats: Dict[str, Dict[str, float]] = {}
  for qid in sorted(item_ids):
    col_name = f"SELF_{qid}"
    if col_name not in df.columns:
      continue
    col = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if col.empty:
      # fallback span
      stats[qid] = {"min": 0.0, "max": 1.0}
    else:
      mn = float(col.min())
      mx = float(col.max())
      if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
        stats[qid] = {"min": 0.0, "max": 1.0}
      else:
        stats[qid] = {"min": mn, "max": mx}
  out_dir = os.path.dirname(out_json)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"items": stats}, f, indent=2)
  return stats


def apply_minmax(df_items: pd.DataFrame, norm: Dict[str, Any]) -> pd.DataFrame:
  X = df_items.copy()
  items = norm.get("items", {})
  for qid, s in items.items():
    col_name = f"SELF_{qid}"
    if col_name not in X.columns:
      continue
    mn = float(s.get("min", 0.0))
    mx = float(s.get("max", 1.0))
    span = mx - mn
    if span <= 0.0:
      span = 1.0
    vals = pd.to_numeric(X[col_name], errors="coerce")
    vals = (vals - mn) / span
    vals = vals.clip(0.0, 1.0)
    X[col_name] = vals
  return X


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--export_csv", required=True)
  ap.add_argument("--meta_json", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  fit_minmax(args.export_csv, args.meta_json, args.out)


if __name__ == "__main__":
  main()

