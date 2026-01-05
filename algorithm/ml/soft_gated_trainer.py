from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Gate:
  w: float = 1.0
  a: float = 1.0
  b: float = 0.0
  mu: float = 0.5
  sigma: float = 0.25


class SoftGatedTrainer:
  """
  learns per domain gates over similarity vs complementarity using pairwise ranking loss
  """

  def __init__(
    self,
    features_npz: str,
    triples: Sequence[Tuple[int, int, int]],
    l1_w: float = 1e-3,
    excl: float = 1e-3,
    seed: int = 0,
  ):
    F = np.load(features_npz)
    self.m = F["m"].astype(np.float32)
    self.dist = F["dist"].astype(np.float32)
    if "rel" in F.files:
      self.rel = F["rel"].astype(np.float32)
    else:
      self.rel = np.ones_like(self.m, dtype=np.float32)
    self.people = [str(p) for p in F["people"]]
    self.domains = [str(d) for d in F["domains"]]
    self.D = len(self.domains)
    self.triples = np.asarray(list(triples), dtype=np.int64)
    self.l1_w = float(l1_w)
    self.excl = float(excl)
    self.rng = np.random.default_rng(seed)
    # parameters per domain
    self.w = np.ones(self.D, dtype=np.float64)
    self.a = np.ones(self.D, dtype=np.float64)
    self.b = np.zeros(self.D, dtype=np.float64)
    self.mu = np.full(self.D, 0.5, dtype=np.float64)
    self.sigma = np.full(self.D, 0.25, dtype=np.float64)

  def _score_idx(self, i: int, j: int):
    m = self.m[i, j, :].astype(np.float64)
    d = self.dist[i, j, :].astype(np.float64)
    r = self.rel[i, j, :].astype(np.float64)
    # similarity and complementarity kernels per domain
    ksim = np.exp(-((d / self.sigma) ** 2))
    kcmp = np.exp(-(((d - self.mu) ** 2) / (2.0 * (self.sigma ** 2))))
    term = self.a * ksim + self.b * kcmp
    score = float(np.sum(self.w * m * term * r))
    return score, m, r, ksim, kcmp

  def _pair_loss(self, s_pos: float, s_neg: float) -> float:
    """
    logistic bpr loss: log(1 + exp(-(s_pos - s_neg)))
    """
    x = float(s_pos - s_neg)
    if x >= 0.0:
      return float(np.log1p(np.exp(-x)))
    # stable branch for large negative x
    return float((-x) + np.log1p(np.exp(-x)))

  def train(self, epochs: int = 50, lr: float = 0.05, log_every: int = 1) -> None:
    if self.triples.size == 0:
      return
    lr = float(lr)
    log_every = max(int(log_every), 1)
    for ep in range(int(epochs)):
      self.rng.shuffle(self.triples)
      epoch_loss = 0.0
      for anchor, pos, neg in self.triples:
        sp, mp, rp, ksim_p, kcmp_p = self._score_idx(int(anchor), int(pos))
        sn, mn, rn, ksim_n, kcmp_n = self._score_idx(int(anchor), int(neg))
        x = sp - sn
        # stable logistic gradient d/dx log(1+exp(-x)) = -1/(1+exp(x))
        if x >= 0:
          g_base = -1.0 / (1.0 + np.exp(x))
        else:
          ex = np.exp(x)
          g_base = -ex / (1.0 + ex)
        # per domain contributions
        term_pos = mp * rp * (self.a * ksim_p + self.b * kcmp_p)
        term_neg = mn * rn * (self.a * ksim_n + self.b * kcmp_n)
        grad_w = g_base * (term_pos - term_neg) + self.l1_w * np.sign(self.w)
        grad_a = g_base * (self.w * (mp * rp * ksim_p - mn * rn * ksim_n)) + self.excl * self.b
        grad_b = g_base * (self.w * (mp * rp * kcmp_p - mn * rn * kcmp_n)) + self.excl * self.a
        self.w -= lr * grad_w
        self.a -= lr * grad_a
        self.b -= lr * grad_b
        # project to non negative region
        self.w = np.clip(self.w, 0.0, None)
        self.a = np.clip(self.a, 0.0, None)
        self.b = np.clip(self.b, 0.0, None)
        epoch_loss += self._pair_loss(sp, sn)
      if (ep + 1) % log_every == 0:
        active = int(np.sum(self.w > 1e-3))
        avg_loss = epoch_loss / max(len(self.triples), 1)
        print(f"[soft] epoch {ep+1:03d}/{int(epochs)}  loss={avg_loss:.4f}  active_domains={active}/{self.D}")

  def infer_modes(self) -> Dict[str, Dict[str, float]]:
    modes: Dict[str, Dict[str, float]] = {}
    sum_w = 0.0
    for idx in range(self.D):
      w = float(self.w[idx])
      if w > 0.0:
        sum_w += w
    scale = 1.0 / sum_w if sum_w > 0.0 else 1.0
    for idx, name in enumerate(self.domains):
      w = float(self.w[idx]) * scale
      a = float(self.a[idx])
      b = float(self.b[idx])
      if w < 1e-3:
        mode = "irrelevant"
      elif a >= 2.0 * b:
        mode = "similarity"
      elif b >= 2.0 * a:
        mode = "complementarity"
      else:
        mode = "mixed"
      modes[name] = {
        "mode": mode,
        "weight": w,
        "mu": float(self.mu[idx]),
        "sigma": float(self.sigma[idx]),
        "alpha": 1.0,
        "beta": 1.0,
        "a": a,
        "b": b,
      }
    return modes

  def to_config(self) -> Dict[str, Dict[str, float]]:
    return self.infer_modes()


def build_triples(features_npz: str, couples_csv: str, neg_k: int = 20, seed: int = 0) -> List[Tuple[int, int, int]]:
  F = np.load(features_npz)
  people = [str(p) for p in F["people"]]
  id_to_idx = {pid: i for i, pid in enumerate(people)}
  couples = pd.read_csv(couples_csv)
  rng = np.random.default_rng(seed)
  triples: List[Tuple[int, int, int]] = []
  all_idx = np.arange(len(people), dtype=int)
  for _, row in couples.iterrows():
    a_id = str(row.get("partner_a_id"))
    b_id = str(row.get("partner_b_id"))
    ia = id_to_idx.get(a_id)
    ib = id_to_idx.get(b_id)
    if ia is None or ib is None:
      continue
    for anchor, partner in ((ia, ib), (ib, ia)):
      cand = [int(x) for x in all_idx if x not in (anchor, partner)]
      if not cand:
        continue
      k = min(int(neg_k), len(cand))
      negs = rng.choice(cand, size=k, replace=False)
      for n in negs:
        triples.append((int(anchor), int(partner), int(n)))
  return triples


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--features", required=True)
  ap.add_argument("--couples_csv", required=True)
  ap.add_argument("--neg_k", type=int, default=20)
  ap.add_argument("--epochs", type=int, default=50)
  ap.add_argument("--lr", type=float, default=0.05)
  ap.add_argument("--l1_w", type=float, default=1e-3)
  ap.add_argument("--excl", type=float, default=1e-3)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--outdir", default="out_soft")
  args = ap.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  triples = build_triples(args.features, args.couples_csv, neg_k=args.neg_k, seed=args.seed)
  trainer = SoftGatedTrainer(args.features, triples, l1_w=args.l1_w, excl=args.excl, seed=args.seed)
  trainer.train(epochs=args.epochs, lr=args.lr)
  cfg = trainer.to_config()
  out_path = os.path.join(args.outdir, "best_config.json")
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
  print(json.dumps({"config_path": out_path, "domains": list(cfg.keys())}, indent=2))


if __name__ == "__main__":
  main()
