from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np


def comp_kernel(delta: np.ndarray, mu: float, sigma: float) -> np.ndarray:
  return np.exp(-((delta - mu) ** 2) / (2.0 * (sigma ** 2)))


def sim_kernel(delta: np.ndarray, sigma: float) -> np.ndarray:
  return np.exp(-((delta / sigma) ** 2))


def make_features(lcbs_npz: str, dists_npz: str, out_npz: str, mus: Sequence[float] = (0.25, 0.5, 0.75), sigmas: Sequence[float] = (0.15, 0.25, 0.35)) -> None:
  L = np.load(lcbs_npz)
  D = len(L["domains"])
  N = len(L["people"])
  lcb = L["lcb"].astype(np.float32)
  m = np.sqrt(lcb[:, :, :, 0].clip(0.0, 1.0) * lcb[:, :, :, 1].clip(0.0, 1.0))
  D2 = np.load(dists_npz)
  dist = D2["dist"].astype(np.float32)
  if dist.shape != m.shape:
    raise SystemExit(f"distance tensor shape {dist.shape} does not match lcb tensor {m.shape}")
  rel = np.ones_like(m, dtype=np.float32)
  out_dir = os.path.dirname(out_npz)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  np.savez_compressed(
    out_npz,
    m=m,
    dist=dist,
    rel=rel,
    people=L["people"],
    domains=L["domains"],
    mus=np.asarray(list(mus), dtype=np.float32),
    sigmas=np.asarray(list(sigmas), dtype=np.float32),
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--lcbs", required=True)
  ap.add_argument("--dists", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  make_features(args.lcbs, args.dists, args.out)


if __name__ == "__main__":
  main()

