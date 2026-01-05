from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Sequence

import numpy as np


@dataclass
class DomainConfig:
  name: str
  weight: float
  mode: str  # 'similarity' | 'complementarity' | 'irrelevant' | 'mixed'
  mu: float = 0.5
  sigma: float = 0.25
  alpha: float = 1.0
  beta: float = 1.0


def k_sim(delta: float, sigma: float) -> float:
  sigma = max(float(sigma), 1e-6)
  return float(np.exp(-((delta / sigma) ** 2)))


def k_comp(delta: float, mu: float, sigma: float) -> float:
  sigma = max(float(sigma), 1e-6)
  return float(np.exp(-(((delta - mu) ** 2) / (2.0 * (sigma ** 2)))))


class ConfigurableScorer:
  """
  scores pairs using precomputed pam mutual acceptabilities and trait distances
  """

  def __init__(self, features_npz: str):
    F = np.load(features_npz)
    self.m = F["m"].astype(np.float32)
    self.dist = F["dist"].astype(np.float32)
    if "rel" in F.files:
      self.rel = F["rel"].astype(np.float32)
    else:
      self.rel = np.ones_like(self.m, dtype=np.float32)
    self.people = [str(p) for p in F["people"]]
    self.domains = [str(d) for d in F["domains"]]
    self._id_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(self.people)}

  def make_domain_configs(self, spec: Dict[str, Any]) -> List[DomainConfig]:
    """
    spec maps domain -> {mode, weight, mu, sigma, alpha, beta}
    """
    cfgs: List[DomainConfig] = []
    for name in self.domains:
      s = spec.get(name, {})
      cfgs.append(
        DomainConfig(
          name=name,
          weight=float(s.get("weight", 0.0)),
          mode=str(s.get("mode", "irrelevant")),
          mu=float(s.get("mu", 0.5)),
          sigma=float(s.get("sigma", 0.25)),
          alpha=float(s.get("alpha", 1.0)),
          beta=float(s.get("beta", 1.0)),
        )
      )
    return cfgs

  def score_pair_idx(self, i: int, j: int, cfgs: Sequence[DomainConfig]) -> float:
    total = 0.0
    for d_idx, dc in enumerate(cfgs):
      if dc.weight <= 0.0 or dc.mode == "irrelevant":
        continue
      m = float(self.m[i, j, d_idx])
      if m <= 0.0:
        continue
      delta = float(self.dist[i, j, d_idx])
      if dc.mode == "similarity":
        k = k_sim(delta, dc.sigma)
      elif dc.mode == "complementarity":
        k = k_comp(delta, dc.mu, dc.sigma)
      else:
        # mixed: average of similarity and complementarity kernels
        k = 0.5 * (k_sim(delta, dc.sigma) + k_comp(delta, dc.mu, dc.sigma))
      total += dc.weight * (m ** dc.alpha) * (k ** dc.beta) * float(self.rel[i, j, d_idx])
    return float(total)

  def score_pair_ids(self, aid: str, bid: str, cfgs: Sequence[DomainConfig]) -> float:
    i = self._id_to_idx.get(str(aid))
    j = self._id_to_idx.get(str(bid))
    if i is None or j is None:
      return 0.0
    return self.score_pair_idx(i, j, cfgs)
