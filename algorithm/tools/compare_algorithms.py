from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Any


BASE_DIR = Path(__file__).resolve().parents[1]  # tiger-alg/
ROOT_DIR = BASE_DIR.parent                     # repo root

COUPLES_CSV = ROOT_DIR / "couples.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open("r", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def _read_metrics_kv(path: Path) -> Dict[str, float]:
  rows = _read_csv(path)
  out: Dict[str, float] = {}
  for r in rows:
    k = r.get("metric")
    v = r.get("value")
    if not k:
      continue
    try:
      out[k] = float(v)
    except Exception:
      continue
  return out


def load_baseline_per_couple(variant: str = "core_c100_n20_geo") -> Dict[str, float]:
  rows = _read_csv(BASE_DIR / "out" / "per_couple_scores.csv")
  by_couple: Dict[str, float] = {}
  for r in rows:
    if r.get("variant") and r.get("variant") != variant:
      continue
    cid = r.get("couple_id") or ""
    if not cid:
      continue
    try:
      by_couple[cid] = float(r.get("score") or "nan")
    except Exception:
      continue
  return by_couple


def load_ml_per_couple(variant: str = "core_c100_n20_geo") -> Dict[str, float]:
  rows = _read_csv(BASE_DIR / "out_ml" / "per_couple_scores.csv")
  by_couple: Dict[str, float] = {}
  for r in rows:
    if r.get("variant") and r.get("variant") != variant:
      continue
    cid = r.get("couple_id") or ""
    if not cid:
      continue
    try:
      by_couple[cid] = float(r.get("score") or "nan")
    except Exception:
      continue
  return by_couple


def load_pam_pct() -> Dict[str, float]:
  couples = _read_csv(COUPLES_CSV)
  pairs = _read_csv(BASE_DIR / "out" / "summary_pam.pairs.csv")
  idx: Dict[tuple[str, str], dict[str, str]] = {}
  for r in pairs:
    a = r.get("A") or ""
    b = r.get("B") or ""
    if not a or not b:
      continue
    idx[(a, b)] = r
    idx[(b, a)] = r
  out: Dict[str, float] = {}
  for c in couples:
    cid = c.get("couple_id") or ""
    a = c.get("partner_a_id") or ""
    b = c.get("partner_b_id") or ""
    if not cid or not a or not b:
      continue
    r = idx.get((a, b))
    if not r:
      continue
    try:
      out[cid] = float(r.get("pct_baseline_like") or "nan")
    except Exception:
      continue
  return out


def _load_pair_scores(path: Path) -> Dict[tuple[str, str], float]:
  rows = _read_csv(path)
  idx: Dict[tuple[str, str], float] = {}
  for r in rows:
    a = r.get("A") or ""
    b = r.get("B") or ""
    s = r.get("score")
    if not a or not b or s is None:
      continue
    try:
      val = float(s)
    except Exception:
      continue
    idx[(a, b)] = val
    idx[(b, a)] = val
  return idx


def load_soft_scores() -> Dict[str, float]:
  couples = _read_csv(COUPLES_CSV)
  idx = _load_pair_scores(BASE_DIR / "out_soft" / "scores_soft.csv")
  out: Dict[str, float] = {}
  for c in couples:
    cid = c.get("couple_id") or ""
    a = c.get("partner_a_id") or ""
    b = c.get("partner_b_id") or ""
    if not cid or not a or not b:
      continue
    val = idx.get((a, b))
    if val is None:
      continue
    out[cid] = val
  return out


def load_evo_scores() -> Dict[str, float]:
  couples = _read_csv(COUPLES_CSV)
  idx = _load_pair_scores(BASE_DIR / "out_evo" / "scores_evo.csv")
  out: Dict[str, float] = {}
  for c in couples:
    cid = c.get("couple_id") or ""
    a = c.get("partner_a_id") or ""
    b = c.get("partner_b_id") or ""
    if not cid or not a or not b:
      continue
    val = idx.get((a, b))
    if val is None:
      continue
    out[cid] = val
  return out


def load_metrics() -> Dict[str, Dict[str, Any]]:
  metrics: Dict[str, Dict[str, Any]] = {}

  # baseline / ML: summary_by_variant.csv
  for label, outdir in (("baseline", "out"), ("ml", "out_ml")):
    path = BASE_DIR / outdir / "summary_by_variant.csv"
    rows = _read_csv(path)
    if not rows:
      continue
    best = rows[0]
    metrics[label] = {
      "auc": float(best.get("auc") or "nan"),
      "lift": float(best.get("lift") or "nan"),
      "median_rank": best.get("median_rank"),
      "n_couples": best.get("n_couples"),
      "true_mean": float(best.get("true_mean") or "nan"),
      "rnd_mean": float(best.get("rnd_mean") or "nan"),
    }

  # pam: summary_pam.csv
  pam_path = BASE_DIR / "out" / "summary_pam.csv"
  if pam_path.exists():
    pam = _read_metrics_kv(pam_path)
    metrics["pam"] = {
      "auc": pam.get("auc"),
      "lift": pam.get("lift"),
      "auc_ci_low": pam.get("auc_ci_low"),
      "auc_ci_high": pam.get("auc_ci_high"),
    }

  # soft / evo: harness metrics.csv
  for label, outdir in (("soft", "out_soft"), ("evo", "out_evo")):
    path = BASE_DIR / outdir / "metrics.csv"
    if not path.exists():
      continue
    m = _read_metrics_kv(path)
    metrics[label] = {
      "auc": m.get("auc"),
      "hit@1": m.get("hit@1"),
      "hit@3": m.get("hit@3"),
      "mutual@3": m.get("mutual@3"),
      "n_anchors": m.get("n_anchors"),
      "n_couples": m.get("n_couples"),
      "mrr": m.get("mrr"),
    }

  return metrics


def main() -> None:
  if not COUPLES_CSV.exists():
    print(f"couples.csv not found at {COUPLES_CSV}")
    return

  couples = _read_csv(COUPLES_CSV)
  baseline = load_baseline_per_couple()
  ml = load_ml_per_couple()
  pam = load_pam_pct()
  soft = load_soft_scores()
  evo = load_evo_scores()
  metrics = load_metrics()

  print("=== Global metrics by algorithm ===")
  order = ["baseline", "ml", "pam", "soft", "evo"]
  for name in order:
    m = metrics.get(name)
    if not m:
      continue
    print(f"\n{name}:")
    for k, v in m.items():
      print(f"  {k}: {v}")

  print("\n=== Per-couple comparison ===")
  header = ["couple", "baseline", "ml", "pam %", "soft", "evo"]
  widths = [8, 10, 10, 8, 10, 10]
  def fmt_row(vals: list[str]) -> str:
    parts = []
    for v, w in zip(vals, widths):
      parts.append(str(v).rjust(w))
    return " ".join(parts)
  print(fmt_row(header))
  print(" ".join("-" * w for w in widths))
  for c in couples:
    cid = c.get("couple_id") or ""
    if not cid:
      continue
    row = [
      cid,
      f"{baseline.get(cid, float('nan')):.4f}" if cid in baseline else "",
      f"{ml.get(cid, float('nan')):.4f}" if cid in ml else "",
      f"{pam.get(cid, float('nan')):.1f}" if cid in pam else "",
      f"{soft.get(cid, float('nan')):.4f}" if cid in soft else "",
      f"{evo.get(cid, float('nan')):.4f}" if cid in evo else "",
    ]
    print(fmt_row(row))


if __name__ == "__main__":
  main()
