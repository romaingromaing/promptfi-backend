# signals/score.py
from __future__ import annotations
from typing import Dict, Any

def score_card(base_prob: float, features: Dict[str, Any]) -> tuple[float, float]:
    prob = base_prob
    vm = float(features.get("vol_mult_20", features.get("vmult", 1.0)))
    adx = float(features.get("adx14", 20.0))
    r1 = float(features.get("r1", 0.3)); r2 = float(features.get("r2", 0.3))
    prob += 0.05 * max(0.0, vm - 1.0)
    prob += 0.03 * max(0.0, (adx - 20)/30)
    prob += 0.02 * min(r1, r2)
    prob = float(min(0.98, max(0.5, prob)))
    confidence = float(min(1.0, 0.5 + 0.2*max(0.0, vm - 1.0) + 0.1*min(r1, r2)))
    return prob, confidence
