from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


IMP_CATS = [0, 1, 10, 50, 250]
IMP_LABELS = {
    "irrelevant": 0,
    "a little": 1,
    "somewhat": 10,
    "very": 50,
    "mandatory": 250,
}


@dataclass
class ImportanceMap:
    m: Dict[int, float]

    @staticmethod
    def from_defaults() -> "ImportanceMap":
        # simple monotone calibration in roughly log scale
        return ImportanceMap({0: 0.0, 1: 1.0, 10: 3.0, 50: 6.0, 250: 10.0})

    def weight(self, c: int | str | None) -> float:
        """Accept ints or human labels; unknowns map to 0."""
        if c is None:
            return 0.0
        try:
            v = int(c)
            return float(self.m.get(v, 0.0))
        except Exception:
            pass
        s = str(c).strip().lower()
        if s in IMP_LABELS:
            return float(self.m.get(IMP_LABELS[s], 0.0))
        return 0.0
