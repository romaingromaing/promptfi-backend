from __future__ import annotations
from typing import List, Dict, Any, Callable
import pandas as pd
# from services.data.ohlcv_intraday import load_ohlcv
from data.ohlcv_intraday import load_ohlcv
from .pivots import recent_pivots
from . import patterns as P
from .breakout import recent_breakout
from .score import score_card

DETECTORS: Dict[str, Callable] = {
    "ascending_triangle": P.ascending_triangle,
    "descending_triangle": P.descending_triangle,
    "symmetrical_triangle": P.symmetrical_triangle,
    "bull_flag": P.bull_flag,
    "bear_flag": P.bear_flag,
    "double_top": P.double_top,
    "double_bottom": P.double_bottom,
    "head_shoulders": P.head_shoulders,
    "inverse_head_shoulders": P.inverse_head_shoulders,
    "wedge_rising": P.wedge_rising,
    "wedge_falling": P.wedge_falling,
    "engulfing_bull": P.engulfing_bull,
    "engulfing_bear": P.engulfing_bear,
    "hammer": P.hammer,
    "shooting_star": P.shooting_star,
    "doji": P.doji,
}

def apply_indicator_filters(df: pd.DataFrame, clauses: List[str]) -> bool:
    c = df["close"]
    env = {
        "CLOSE": c.iloc[-1],
        "SMA30": c.rolling(30).mean().iloc[-1],
        "EMA50": c.ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1],
        "EMA200": c.ewm(span=200, adjust=False, min_periods=200).mean().iloc[-1],
        "RSI14": _rsi(c, 14),
        "V": df["volume"].iloc[-1],
        "VOL_SMA20": df["volume"].rolling(20).mean().iloc[-1]
    }
    for expr in clauses or []:
        e = expr.upper().replace(" ", "")
        if e.startswith("RSI(14)"): e = e.replace("RSI(14)", "RSI14")
        if ">" in e:
            lhs, rhs = e.split(">")
            try_rhs = float(rhs) if rhs.replace(".","",1).isdigit() else env.get(rhs)
            if env.get(lhs, 0) <= (try_rhs or 0): return False
        elif "<" in e:
            lhs, rhs = e.split("<")
            try_rhs = float(rhs) if rhs.replace(".","",1).isdigit() else env.get(rhs)
            if env.get(lhs, 0) >= (try_rhs or 0): return False
    return True

def _rsi(close: pd.Series, n=14) -> float:
    d = close.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au/(ad+1e-12)
    return float(100 - (100/(1+rs)).iloc[-1])

def scan(symbols: List[str], tf: str, patterns: List[str],
         indicator_filters: List[str] | None = None,
         recent_breakout_flag: bool = False,
         recency_bars: int = 5,
         bars: int = 720,
         sort: str = "prob",
         limit: int = 12) -> Dict[str, Any]:

    cards: List[Dict[str, Any]] = []
    for sym in symbols:
        df = load_ohlcv(sym, timeframe=tf, bars=bars)
        if len(df) < 50: continue
        if indicator_filters and not apply_indicator_filters(df, indicator_filters):
            continue
        pivs = recent_pivots(df, tf=tf)
        for patt in patterns:
            det = DETECTORS[patt](df, pivs, tf, sym)
            if det.matched and det.card:
                if recent_breakout_flag:
                    rb = recent_breakout(df, lookback=20, confirm_bars=recency_bars, long=("bear" not in patt and "head_shoulders" not in patt))
                    if not rb: 
                        continue
                    det.card["features"]["recent_breakout"] = True
                p, conf = score_card(det.card.get("prob", 0.55), det.card.get("features", {}))
                det.card["prob"] = p; det.card["confidence"] = conf
                cards.append(det.card)
    cards.sort(key=lambda x: x.get("prob", 0.0), reverse=True if sort=="prob" else False)
    return {"cards": cards[:limit]}
