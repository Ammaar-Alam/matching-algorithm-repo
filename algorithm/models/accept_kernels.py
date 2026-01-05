from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp
from typing import Dict, Any, List, Tuple


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0/(1.0+z)
    else:
        z = exp(x)
        return z/(1.0+z)


def _tol_from_label(lbl: str) -> int:
    m = {"Same answer": 0, "±1 step": 1, "±2 steps": 2, "Any answer": 3}
    return m.get((lbl or "±1 step").strip(), 1)


@dataclass
class LikertParams:
    alpha: float
    a_steps: List[float]  # len 3, non‑positive, cumulative distance penalties
    b_steps: List[float]  # len 3, non‑negative, cumulative tolerance bonuses


@dataclass
class MCParams:
    alpha: float
    eta: float  # non‑negative, slope in similarity


@dataclass
class AcceptabilityParams:
    """per‑item kernels; keys are item ids like 'Q1'"""
    likert: Dict[str, LikertParams]
    mc: Dict[str, MCParams]

    @staticmethod
    def from_defaults(meta: Dict[str, Dict[str, Any]]) -> "AcceptabilityParams":
        likert: Dict[str, LikertParams] = {}
        mc: Dict[str, MCParams] = {}
        for qid, m in meta.items():
            rt = str(m.get("ResponseType", "Likert5"))
            if rt.startswith("Likert"):
                # fairly permissive defaults, monotone by construction
                likert[qid] = LikertParams(
                    alpha=2.0,
                    a_steps=[-1.5, -1.0, -0.7],
                    b_steps=[0.8, 0.6, 0.4],
                )
            elif rt.startswith("MC"):
                mc[qid] = MCParams(alpha=-1.0, eta=3.0)
        return AcceptabilityParams(likert=likert, mc=mc)

    @staticmethod
    def load(path: str) -> "AcceptabilityParams":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        likert = {}
        mc = {}
        for qid, p in obj.get("likert", {}).items():
            likert[qid] = LikertParams(alpha=float(p["alpha"]),
                                       a_steps=[float(x) for x in p["a_steps"]],
                                       b_steps=[float(x) for x in p["b_steps"]])
        for qid, p in obj.get("mc", {}).items():
            mc[qid] = MCParams(alpha=float(p["alpha"]), eta=float(p["eta"]))
        return AcceptabilityParams(likert=likert, mc=mc)

    def dump(self, path: str) -> None:
        out = {
            "likert": { qid: {"alpha":p.alpha, "a_steps":p.a_steps, "b_steps":p.b_steps}
                         for qid, p in self.likert.items() },
            "mc": { qid: {"alpha":p.alpha, "eta":p.eta} for qid, p in self.mc.items() },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


def p_likert(params: LikertParams, self_a: str, self_b: str, tol_label: str) -> float:
    try:
        a = int(self_a)
        b = int(self_b)
    except Exception:
        return 0.0
    d = abs(a - b)
    if d > 3:
        d = 3
    t = _tol_from_label(tol_label)
    if t > 3:
        t = 3
    # logits with cumulative step functions to enforce monotonicity by sign of steps
    # a_steps are <=0 and applied when d>=c; b_steps are >=0 and applied when t>=r
    logit = params.alpha
    a = params.a_steps
    b = params.b_steps
    if d >= 1:
        logit += a[0]
    if d >= 2:
        logit += a[1]
    if d >= 3:
        logit += a[2]
    if t >= 1:
        logit += b[0]
    if t >= 2:
        logit += b[1]
    if t >= 3:
        logit += b[2]
    return _sigmoid(float(logit))


def p_mc(params: MCParams, sim_to_set: float) -> float:
    # similarity already in [0,1]
    logit = params.alpha + params.eta * float(sim_to_set)
    return _sigmoid(logit)


