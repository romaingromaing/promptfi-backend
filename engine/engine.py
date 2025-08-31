# import pandas as pd, numpy as np
# from typing import Dict, List, Tuple
# from .indicators import SMA, EMA, RSI, VOL_SMA, RET_60D
# from .gates import evaluate_gate

# REB_FREQ = "W-FRI"

# def build_features(ohlcv: Dict[str,pd.DataFrame], feats: List[str]) -> Dict[str,pd.DataFrame]:
#     out = {}
#     for a,df in ohlcv.items():
#         x = df.copy()
#         if any(f.startswith("SMA_") for f in feats):
#             for f in [f for f in feats if f.startswith("SMA_")]:
#                 n = int(f.split("_")[1]); x[f] = SMA(x["close"], n)
#         if any(f.startswith("EMA_") for f in feats):
#             for f in [f for f in feats if f.startswith("EMA_")]:
#                 n = int(f.split("_")[1]); x[f] = EMA(x["close"], n)
#         if "RSI_14" in feats:
#             x["RSI_14"] = RSI(x["close"], 14)
#         if any(f.startswith("VOL_SMA_") for f in feats):
#             for f in [f for f in feats if f.startswith("VOL_SMA_")]:
#                 n = int(f.split("_")[2]); x[f] = VOL_SMA(x["volume"], n)
#         if "RET_60D" in feats:
#             x["RET_60D"] = RET_60D(x["close"])
#         out[a] = x
#     return out

# def combine_gates(plan, feats: Dict[str,pd.DataFrame], sentiment: Dict[str,pd.Series]) -> Dict[str,pd.Series]:
#     all_of = plan.get("gates",{}).get("all_of",[])
#     any_of = plan.get("gates",{}).get("any_of",[])
#     elig = {}
#     for a,df in feats.items():
#         s = pd.Series(True, index=df.index)
#         for g in all_of:
#             s &= evaluate_gate(g["expr"], df, sentiment.get(a))
#         if any_of:
#             s_any = pd.Series(False, index=df.index)
#             for g in any_of:
#                 s_any |= evaluate_gate(g["expr"], df, sentiment.get(a))
#             s &= s_any
#         elig[a] = s.fillna(False)
#     return elig

# def composite_scores(plan, feats: Dict[str,pd.DataFrame], sentiment: Dict[str,pd.Series]) -> Dict[str,pd.Series]:
#     # Example blend: 0.4*trend + 0.3*momentum(RET_60D) + 0.2*volume_strength + 0.1*sentiment
#     # Trend proxy: CLOSE/SMA_30 - 1 (when SMA_30 exists)
#     out = {}
#     for a,df in feats.items():
#         trend = (df["close"]/df.get("SMA_30", df["close"])).fillna(1.0) - 1.0
#         mom   = df.get("RET_60D", pd.Series(0.0, index=df.index))
#         volx  = (df["volume"]/df.get("VOL_SMA_30", df["volume"])).fillna(1.0) - 1.0
#         sent  = sentiment.get(a).reindex(df.index).fillna(method="ffill").fillna(0.0) if a in sentiment else pd.Series(0.0, index=df.index)

#         s = (0.4*trend.clip(lower=0) +
#              0.3*mom.clip(lower=0) +
#              0.2*volx.clip(lower=0) +
#              0.1*sent.clip(lower=0))  # long-only composite
#         out[a] = s.clip(lower=0.0)
#     return out

# def rebalance_dates(idx: pd.DatetimeIndex, cadence="weekly") -> pd.DatetimeIndex:
#     if cadence=="weekly": return pd.date_range(idx[0], idx[-1], freq=REB_FREQ)
#     return pd.date_range(idx[0], idx[-1], freq=REB_FREQ)

# def weights_from_scores(scores_on_date: Dict[str,float], max_w=0.40, hard_cap=0.50) -> Dict[str,float]:
#     # normalize then cap
#     tot = sum(scores_on_date.values()) or 1.0
#     w = {a: (v/tot) for a,v in scores_on_date.items()}
#     # max cap
#     w = {a: min(v, max_w) for a,v in w.items()}
#     s = sum(w.values())
#     w = {a: v/s for a,v in w.items()}
#     # hard cap sanity
#     w = {a: min(v, hard_cap) for a,v in w.items()}
#     s = sum(w.values()); w = {a: v/s for a,v in w.items()}
#     return w

# def plan_trades(current_w: Dict[str,float],
#                 target_w: Dict[str,float],
#                 port_value: float,
#                 band_pp=5.0,
#                 turnover_max=0.15,
#                 order_chunk_usd=2000,
#                 slippage_bps=80) -> Dict:
#     # Only move if |diff| > band_pp; move halfway
#     to_dollars, turnover = {}, 0.0
#     for a,tw in target_w.items():
#         cw = current_w.get(a, 0.0)
#         diff = tw - cw
#         if abs(diff) > (band_pp/100.0):
#             adj = diff * 0.5
#             dollars = adj * port_value
#             to_dollars[a] = dollars
#             turnover += abs(dollars)
#     # scale to turnover cap
#     if turnover > (turnover_max*port_value) and turnover > 0:
#         scale = (turnover_max*port_value)/turnover
#         to_dollars = {a: v*scale for a,v in to_dollars.items()}
#     # chunk orders
#     trade_plan = []
#     for a,usd in to_dollars.items():
#         side = "BUY" if usd>0 else "SELL"
#         chunks, remain = [], abs(usd)
#         while remain > 0:
#             c = min(order_chunk_usd, remain)
#             chunks.append(c); remain -= c
#         trade_plan.append({"asset":a,"side":side,"usd":abs(usd),"chunks":chunks,"slippage_bps":slippage_bps})
#     return {"orders": trade_plan}

# def run_engine(plan: Dict, ohlcv: Dict[str,pd.DataFrame], sentiment: Dict[str,pd.Series]) -> Dict:
#     assets = plan["universe"]
#     feats_needed = ["SMA_30","VOL_SMA_30","RET_60D","RSI_14","EMA_50","SENTIMENT"]  # safe superset for demo
#     feats = build_features({a: ohlcv[a] for a in assets}, feats_needed)
#     elig = combine_gates(plan, feats, sentiment)
#     scores = composite_scores(plan, feats, sentiment)

#     # portfolio loop
#     idx = list(feats.values())[0].index
#     rdates = set(rebalance_dates(idx, cadence=plan["rebalance"]["cadence"]))
#     target_weights_by_date = {}
#     explain = {}
#     for t in idx:
#         if t not in rdates: continue
#         # candidates
#         cands = [a for a in assets if bool(elig[a].get(t, False))]
#         if not cands:
#             target_weights_by_date[t] = {a: 0.0 for a in assets}
#             continue
#         sc = {a: float(scores[a].get(t,0.0)) for a in cands}
#         target_weights_by_date[t] = weights_from_scores(sc, max_w=plan["risk"]["max_weight"], hard_cap=plan["rebalance"]["hard_cap"])
#         # Explain bullets
#         explain[t] = {}
#         for a in assets:
#             bullets = []
#             df = feats[a]
#             if "SMA_30" in df.columns:
#                 sma30 = df.loc[:t,"SMA_30"].dropna().iloc[-1] if t in df.index else None
#                 px    = df.loc[:t,"close"].dropna().iloc[-1] if t in df.index else None
#                 if sma30 and px:
#                     bullets.append(f"price {((px/sma30)-1)*100:.1f}% > SMA30" if px>sma30 else f"price {((px/sma30)-1)*100:.1f}% < SMA30")
#             if "VOL_SMA_30" in df.columns:
#                 vsma = df.loc[:t,"VOL_SMA_30"].dropna().iloc[-1] if t in df.index else None
#                 vol  = df.loc[:t,"volume"].dropna().iloc[-1] if t in df.index else None
#                 if vsma and vol:
#                     bullets.append(f"vol {(vol/(vsma+1e-9)):.2f}× avg")
#             srow = sentiment.get(a, pd.Series([0], index=[t])).loc[:t].iloc[-1] if a in sentiment else 0.0
#             bullets.append(f"sent {srow:+.2f}")
#             explain[t][a] = bullets
#     # For live trading, current_w will come from vault; here assume cash start
#     # Trade plan is created in backtest/live using plan_trades(...)
#     return {
#         "eligibility": {a: elig[a] for a in assets},
#         "scores": {a: scores[a] for a in assets},
#         "target_weights": target_weights_by_date,
#         "explain_metrics": explain
#     }


# engine.py
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

from engine.indicators import (
    sma, ema, rsi, stoch_rsi, macd, atr, adx, di_plus_minus,
    donchian_high, donchian_low, bollinger, keltner, squeeze_bb_kc,
    vwap, obv, cmf, mfi, support_levels, volume_profile_nodes, rsi_divergence
)
from services.planner.regime import classify_regime, map_regime_to_template

# ---------- Plan dataclass ----------
@dataclass
class Plan:
    regime: str
    direction_bias: str
    universe: List[str]
    gates: dict
    custom_rules: List[str]
    weighting: dict
    rebalance: dict
    risk: dict
    execution: dict
    sentiment_cfg: dict

# ---------- Helpers ----------
def _sent_val(s_series: pd.Series | None) -> float:
    if s_series is None or s_series.empty: return 0.0
    return float(s_series.iloc[-1])

def _bool_adx_rising(adx_series: pd.Series) -> bool:
    return (adx_series.diff().tail(5) > 0).sum() >= 3

# ---------- Rule parser (custom_rules) ----------
_RULES = [
    (re.compile(r"CLOSE\s*>\s*SMA\((\d+)\)", re.I), lambda df,m: df["close"].iloc[-1] > sma(df["close"], int(m)).iloc[-1]),
    (re.compile(r"CLOSE\s*>\s*EMA\((\d+)\)", re.I), lambda df,m: df["close"].iloc[-1] > ema(df["close"], int(m)).iloc[-1]),
    (re.compile(r"RSI\((\d+)\)\s*>\s*(\d+)", re.I), lambda df,n,t: rsi(df["close"], int(n)).iloc[-1] > int(t)),
    (re.compile(r"RSI\((\d+)\)\s*<\s*(\d+)", re.I), lambda df,n,t: rsi(df["close"], int(n)).iloc[-1] < int(t)),
    (re.compile(r"VOLUME\s*>\s*([0-9.]+)\*VOL_SMA\((\d+)\)", re.I),
        lambda df,k,n: df["volume"].iloc[-1] > float(k)*df["volume"].rolling(int(n)).mean().iloc[-1]),
    (re.compile(r"SENTIMENT\s*>=\s*GOOD", re.I), lambda df: True),
]

def check_custom_rules(df: pd.DataFrame, rules: List[str], sentiment_ok: bool) -> bool:
    for rtxt in rules:
        ok = None
        for rx, fn in _RULES:
            m = rx.match(rtxt.replace(" ", ""))
            if m:
                try:
                    parsed = []
                    for g in m.groups():
                        try:
                            parsed.append(int(g))
                        except ValueError:
                            parsed.append(float(g))
                    ok = fn(df, *parsed)
                except TypeError:
                    ok = fn(df, *m.groups())
                break
        if ok is None and "SENTIMENT" in rtxt.upper():
            ok = sentiment_ok
        if ok is False:
            return False
    return True

# ---------- Scoring ----------
def trend_score(df: pd.DataFrame) -> float:
    c,h,l = df["close"], df["high"], df["low"]
    adx14 = adx(h,l,c,14).iloc[-1]
    ema50 = ema(c,50).iloc[-1]; ema200 = ema(c,200).iloc[-1]
    slope = (ema(c,50).iloc[-1] / (ema(c,50).shift(5).iloc[-1] + 1e-9)) - 1.0
    t = 0.5*(1 + math.tanh(5*slope))
    a = min(max((adx14-10)/30, 0), 1)
    bias = 0.2 if ema50 > ema200 else 0.0
    return max(0.0, min(1.0, 0.5*t + 0.5*a + bias))

def momentum_score(df: pd.DataFrame) -> float:
    c = df["close"]
    if len(c)<60: return 0.0
    r60 = c.iloc[-1]/(c.shift(60).iloc[-1]+1e-9)-1.0
    return max(0.0, min(1.0, (r60 + 0.5) / 2.0))

def volume_score(df: pd.DataFrame) -> float:
    v = df["volume"]; vs = v.rolling(20).mean().iloc[-1]
    if vs<=0: return 0.0
    mult = v.iloc[-1]/vs
    return max(0.0, min(1.0, (mult-1.0)/1.5))

def breakout_score(df: pd.DataFrame) -> float:
    c,h,l = df["close"], df["high"], df["low"]
    dhi = donchian_high(h,20).iloc[-1]; a = atr(h,l,c,14).iloc[-1]
    return max(0.0, min(1.0, (c.iloc[-1]-dhi)/(a+1e-9)))

def composite_score(df: pd.DataFrame, coeffs: dict, sent_val: float, template: str) -> float:
    sc_trend = trend_score(df)
    sc_momo  = momentum_score(df)
    sc_vol   = volume_score(df)
    sc_extra = 0.0
    if template == "breakout_up":
        sc_extra = 0.5*breakout_score(df)
    s = (coeffs.get("trend",0.35)*sc_trend +
         coeffs.get("momentum",0.35)*sc_momo +
         coeffs.get("volume",0.15)*sc_vol +
         coeffs.get("sentiment",0.15)*max(0.0, sent_val) +
         sc_extra)
    return max(0.0, min(1.0, s))

# ---------- Strategy selection (auto) ----------
def infer_plan_template(plan: Plan, ohlcv: Dict[str,pd.DataFrame]) -> str:
    if plan.regime != "auto":
        return {"trend":"trend_follow","range":"mean_revert","breakout":"breakout_up","support":"support_bounce"}.get(plan.regime,"trend_follow")
    votes = {}
    for df in ohlcv.values():
        r = classify_regime(df)
        votes[r] = votes.get(r,0)+1
    reg = max(votes.items(), key=lambda x:x[1])[0] if votes else "other"
    return map_regime_to_template(reg)

# ---------- Weights & Explain ----------
def target_weights(plan: Plan, ohlcv: Dict[str,pd.DataFrame], sentiment: Dict[str,pd.Series]):
    template = infer_plan_template(plan, ohlcv)
    coeffs   = plan.weighting.get("coeffs", {"trend":0.35,"momentum":0.35,"volume":0.15,"sentiment":0.15})
    good     = plan.sentiment_cfg.get("good_threshold", 0.30)
    bad      = plan.sentiment_cfg.get("bad_threshold", -0.30)
    tilt_pct = plan.weighting.get("tilt_sentiment_pct", 0.10)

    weights = {}; explains = {}
    for a, df in ohlcv.items():
        s_val = _sent_val(sentiment.get(a))
        sentiment_ok = (s_val >= good) if plan.gates.get("sentiment","AUTO") in ("AUTO","GOOD") else True

        # Apply custom_rules (if any)
        if plan.custom_rules:
            if not check_custom_rules(df, plan.custom_rules, sentiment_ok):
                continue  # asset gated out

        base = composite_score(df, coeffs, s_val, template)

        # Direction bias
        if plan.direction_bias == "bullish":
            base = min(1.0, base * 1.15)
        elif plan.direction_bias == "bearish":
            if base < 0.7:
                base *= 0.25
            else:
                base *= 0.5

        # Sentiment tilt
        tilt = 0.0
        if s_val >= good: tilt = tilt_pct * min(1.0, s_val)
        elif s_val <= bad: tilt = -tilt_pct * min(1.0, abs(s_val))
        score = max(0.0, base * (1 + tilt))

        weights[a] = score

        # Explain bullets
        c,h,l,v = df["close"], df["high"], df["low"], df["volume"]
        ema50, ema200 = ema(c,50).iloc[-1], ema(c,200).iloc[-1]
        adx14 = adx(h,l,c,14).iloc[-1]
        rsi14 = rsi(c,14).iloc[-1]
        m, bb_u, bb_l, bw, pctb = bollinger(c,20,2)
        kc_m, kc_u, kc_l = keltner(h,l,c,20,1.5)
        squeeze = "ON" if ((bb_u.iloc[-1] < kc_u.iloc[-1]) and (bb_l.iloc[-1] > kc_l.iloc[-1])) else "OFF"
        div = rsi_divergence(c, 100, 5)
        hvns = volume_profile_nodes(c.tail(180), v.tail(180))
        sup  = support_levels(h,l,c,180,5,0.75)

        explains[a] = {
            "template": template,
            "trend": f"EMA50 {'>' if ema50>ema200 else '<'} EMA200, ADX14={adx14:.1f}",
            "momentum_60d": f"{(c.iloc[-1]/(c.shift(60).iloc[-1] + 1e-9) - 1.0):.2%}" if len(c)>60 else "n/a",
            "rsi14": f"{rsi14:.1f}",
            "bb_bw": f"{bw.iloc[-1]:.3f}", "bb_squeeze": squeeze, "pctB": f"{pctb.iloc[-1]:.2f}",
            "divergence": div,
            "hvns": [round(x,2) for x in hvns],
            "supports": [round(x,2) for x in sup],
            "sentiment": f"{s_val:+.2f}",
            "score": f"{score:.3f}"
        }

    # Normalize with caps
    if not weights or sum(weights.values())<=0:
        return {}, explains
    cap  = plan.risk.get("max_weight", 0.40)
    hard = plan.risk.get("hard_cap", 0.50)
    w = {k: min(v, cap) for k,v in weights.items()}
    s = sum(w.values()); w = {k: v/s for k,v in w.items()} if s>0 else {}
    changed = True
    while changed and w:
        changed=False; over=[k for k,v in w.items() if v>hard]
        if over:
            changed=True; excess=sum(w[k]-hard for k in over)
            for k in over: w[k]=hard
            rem=[k for k in w if k not in over]; rs=sum(w[k] for k in rem)
            for k in rem:
                w[k] = w[k] + (w[k]/(rs+1e-9))*excess if rs>0 else w[k]
    s = sum(w.values()); w = {k: v/s for k,v in w.items()} if s>0 else {}
    return w, explains

def build_trade_plan(current_weights: Dict[str,float], target_weights_: Dict[str,float], nav_usd: float, band_pp: float):
    plan = {}
    for a, tw in target_weights_.items():
        cw = current_weights.get(a, 0.0)
        diff = tw - cw
        if abs(diff) > band_pp/100.0:
            plan[a] = (diff * 0.5) * nav_usd   # move halfway
    return plan
