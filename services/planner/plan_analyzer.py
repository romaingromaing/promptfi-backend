import re
from typing import Dict, List

MAX_LOOKBACK = 540

def analyze_plan(plan: Dict) -> Dict:
    assets = plan["universe"]
    feats: List[str] = []

    def add_feat(tok: str):
        if tok not in feats: feats.append(tok)

    patt_sma = re.compile(r"SMA\((\d+)(?:,VOLUME)?\)")
    patt_ema = re.compile(r"EMA\((\d+)\)")
    gates = plan.get("gates", {}).get("all_of", []) + plan.get("gates", {}).get("any_of", [])
    for g in gates:
        expr = g["expr"].upper()
        if "SMA(" in expr:
            for n in patt_sma.findall(expr):
                if ",VOLUME" in expr: add_feat(f"VOL_SMA_{n}")
                else: add_feat(f"SMA_{n}")
        if "EMA(" in expr:
            for n in patt_ema.findall(expr):
                add_feat(f"EMA_{n}")
        if "RSI(14)" in expr: add_feat("RSI_14")
        if "RET_60D" in expr: add_feat("RET_60D")
        if "SENTIMENT" in expr: add_feat("SENTIMENT")

    # derive lookback: max window + 20% buffer (min 90)
    windows = []
    for f in feats:
        if f.startswith("SMA_") or f.startswith("EMA_") or f.startswith("VOL_SMA_"):
            windows.append(int(f.split("_")[-1]))
        if f == "RET_60D": windows.append(60)
        if f == "RSI_14": windows.append(14)
    need = max(windows) if windows else 60
    need = int(min(MAX_LOOKBACK, max(90, need * 1.2)))

    return {"assets": assets, "features": feats, "lookback_days": need}
