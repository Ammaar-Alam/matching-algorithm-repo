from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np

# ensure tiger-alg root on path for sibling packages
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from eval.harness import _compute_metrics


def _random_config(domains: List[str], mus: List[float], sigmas: List[float], rng: np.random.Generator) -> Dict[str, Any]:
  cfg: Dict[str, Any] = {}
  for d in domains:
    mode = rng.choice(["similarity", "complementarity", "irrelevant"])
    weight = float(rng.uniform(0.0, 5.0))
    mu = float(rng.choice(mus))
    sigma = float(rng.choice(sigmas))
    cfg[d] = {"mode": mode, "weight": weight, "mu": mu, "sigma": sigma, "alpha": 1.0, "beta": 1.0}
  return cfg


def _objective(metrics: Dict[str, float], topk: int) -> float:
  def nz(x: float) -> float:
    return x if x == x else 0.0
  h1 = nz(metrics.get("hit@1", float("nan")))
  hk = nz(metrics.get(f"hit@{topk}", float("nan")))
  auc = nz(metrics.get("auc", float("nan")))
  mutual = nz(metrics.get(f"mutual@{topk}", float("nan")))
  # emphasize correct top-1 identification and mutual@K, then hit@K, with auc as a softer tie-break
  return 3.0 * h1 + 2.0 * mutual + 1.0 * hk + 0.5 * auc


def evolutionary_search(
  features_npz: str,
  couples_csv: str,
  mus: List[float],
  sigmas: List[float],
  pop_size: int = 80,
  generations: int = 100,
  topk: int = 3,
  seed_cfg: Dict[str, Any] | None = None,
  seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
  F = np.load(features_npz)
  domains = [str(d) for d in F["domains"]]
  rng = np.random.default_rng(seed)

  population: List[Dict[str, Any]] = []
  if seed_cfg is not None:
    population.append(json.loads(json.dumps(seed_cfg)))
  while len(population) < pop_size:
    population.append(_random_config(domains, mus, sigmas, rng))

  history: List[Dict[str, Any]] = []
  best_cfg: Dict[str, Any] = population[0]
  best_score = float("-inf")

  for gen in range(int(generations)):
    scored: List[Tuple[float, Dict[str, Any], Dict[str, float]]] = []
    for cfg in population:
      metrics, _ = _compute_metrics(features_npz, couples_csv, cfg, topk=topk, scores_out=None)
      score = _objective(metrics, topk)
      scored.append((score, cfg, metrics))
      if score > best_score:
        best_score = score
        best_cfg = json.loads(json.dumps(cfg))
      history.append(
        {
          "generation": gen,
          "hit@1": metrics.get("hit@1", float("nan")),
          f"hit@{topk}": metrics.get(f"hit@{topk}", float("nan")),
          "auc": metrics.get("auc", float("nan")),
          "objective": score,
        }
      )
    scored.sort(key=lambda t: t[0], reverse=True)
    # progress logging for the current generation
    best_gen_score, _, best_gen_metrics = scored[0]
    print(
      f"[evo] gen {gen+1:03d}/{int(generations)}  "
      f"obj={best_gen_score:.3f}  "
      f"hit@1={best_gen_metrics.get('hit@1', float('nan')):.3f}  "
      f"hit@{topk}={best_gen_metrics.get(f'hit@{topk}', float('nan')):.3f}  "
      f"auc={best_gen_metrics.get('auc', float('nan')):.3f}"
    )
    elites = [cfg for _, cfg, _ in scored[: max(1, int(0.2 * pop_size))]]

    # build next generation
    next_pop: List[Dict[str, Any]] = []
    next_pop.extend(elites)
    # crossover and mutation
    while len(next_pop) < pop_size:
      if rng.random() < 0.1:
        # inject random
        next_pop.append(_random_config(domains, mus, sigmas, rng))
        continue
      p1, p2 = rng.choice(elites, size=2, replace=True)
      child: Dict[str, Any] = {}
      for d in domains:
        g1 = p1.get(d, {})
        g2 = p2.get(d, {})
        base = g1 if rng.random() < 0.5 else g2
        mode = base.get("mode", "irrelevant")
        weight = float(base.get("weight", 0.0))
        mu = float(base.get("mu", rng.choice(mus)))
        sigma = float(base.get("sigma", rng.choice(sigmas)))
        # small mutations
        if rng.random() < 0.3:
          weight = float(np.clip(weight + rng.uniform(-0.3, 0.3), 0.0, 5.0))
        if rng.random() < 0.1:
          mode = rng.choice(["similarity", "complementarity", "irrelevant"])
        child[d] = {"mode": mode, "weight": weight, "mu": mu, "sigma": sigma, "alpha": 1.0, "beta": 1.0}
      next_pop.append(child)
    population = next_pop

  return best_cfg, history


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--features", required=True)
  ap.add_argument("--couples_csv", required=True)
  ap.add_argument("--pop", type=int, default=80)
  ap.add_argument("--gens", type=int, default=120)
  ap.add_argument("--topk", type=int, default=3)
  ap.add_argument("--seed_config", default=None, help="optional path to seed config json (e.g., out_soft/best_config.json)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--outdir", default="out_evo")
  args = ap.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  F = np.load(args.features)
  if "mus" in F.files:
    mus_arr = F["mus"]
  else:
    mus_arr = np.array([0.25, 0.5, 0.75], dtype=float)
  if "sigmas" in F.files:
    sigmas_arr = F["sigmas"]
  else:
    sigmas_arr = np.array([0.15, 0.25, 0.35], dtype=float)
  mus = [float(x) for x in mus_arr]
  sigmas = [float(x) for x in sigmas_arr]

  seed_cfg: Dict[str, Any] | None = None
  if args.seed_config:
    with open(args.seed_config, "r", encoding="utf-8") as f:
      seed_cfg = json.load(f)

  best_cfg, history = evolutionary_search(
    features_npz=args.features,
    couples_csv=args.couples_csv,
    mus=mus,
    sigmas=sigmas,
    pop_size=args.pop,
    generations=args.gens,
    topk=args.topk,
    seed_cfg=seed_cfg,
    seed=args.seed,
  )

  # normalize weights for readability
  sum_w = 0.0
  for d, cfg_d in best_cfg.items():
    w = float(cfg_d.get("weight", 0.0))
    if w > 0.0:
      sum_w += w
  if sum_w > 0.0:
    for d, cfg_d in best_cfg.items():
      w = float(cfg_d.get("weight", 0.0))
      cfg_d["weight"] = float(w / sum_w) if w > 0.0 else 0.0

  best_path = os.path.join(args.outdir, "best_config.json")
  with open(best_path, "w", encoding="utf-8") as f:
    json.dump(best_cfg, f, indent=2)

  hist_path = os.path.join(args.outdir, "history.csv")
  import pandas as pd

  pd.DataFrame(history).to_csv(hist_path, index=False)

  print(json.dumps({"best_config": best_path, "history": hist_path}, indent=2))


if __name__ == "__main__":
  main()
