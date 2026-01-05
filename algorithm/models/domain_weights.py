from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set
from algorithms.core import DEFAULT_DOMAIN_MULTIPLIER


@dataclass
class DomainWeights:
    m: Dict[str, float]
    psi: float = 1.5

    @staticmethod
    def from_defaults() -> "DomainWeights":
        return DomainWeights(m=dict(DEFAULT_DOMAIN_MULTIPLIER), psi=1.5)

    def weight(self, domain: str, top3: Set[str] | None = None) -> float:
        w = float(self.m.get(domain, 0.0))
        if w == 0.0:
            return 0.0
        if top3 and domain in top3:
            return w * float(self.psi)
        return w

